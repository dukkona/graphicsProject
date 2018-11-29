//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Dukkon Adam	
// Neptun : WJIOZE
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...
static const int ESC = 27; //esc
static const float EPS = 0.001f;


// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

unsigned int shaderProgram; // handle of the shader program
const unsigned int TextureSize = 256;
long last_time = 0;


void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}


// vertex shader in GLSL
// Per-pixel shading - Vertex shader -> bmeinkr 40. dia
const char *vertexSource = R"(
#version 130
precision highp float;

uniform mat4 M, Minv, MVP;		// MVP, Model, Model-inverse
uniform vec4 wEye;				// pos of eye

uniform vec4 light_pos;			// pos of light source 1
uniform vec4 light_pos2;		// pos of light source 2

in vec4 vtxPos;					// pos in modeling space
in vec4 vtxNorm;				// normal in modeling space
in vec2 vtxUV;					// Texture -> bmeinkr 55. dia

out vec3 wNormal;				// normal in world space
out vec3 wView;					// view in world space
out vec3 wLight;				// light dir in world space
out vec3 wLight2;				// light2 dir in world space
out vec2 texcoord;				// Texture -> bmeinkr 55. dia

void main() {
    
    gl_Position = vec4(vtxPos.xyz, 1) * MVP;
    texcoord = vtxUV;						// Texture -> bmeinkr 55. dia		
    vec4 wPos = vec4(vtxPos.xyz, 1) * M;
    
    wLight  =	light_pos.xyz * wPos.w - wPos.xyz * light_pos.w;
    wLight2  =	light_pos2.xyz * wPos.w - wPos.xyz * light_pos2.w;
    
    wView   = wEye.xyz * wPos.w - wPos.xyz;
    wNormal = (Minv * vec4(vtxNorm.xyz, 0)).xyz;
    
})";

// fragment shader in GLSL
// Per-pixel shading - Pixel shader -> bmeinkr 41. dia
const char *fragmentSource = R"(
#version 130
precision highp float;

uniform sampler2D samplerUnit;		// Texture -> bmeinkr 55. dia	

uniform vec4 La;					//ambient
uniform vec4 ks;					//spekular


uniform vec4 Le;					//point source 1
uniform vec4 Le2;					//point source 2

in vec3 wNormal;					// interpolated world sp normal
in vec3 wView;						// interpolated world sp view
in vec3 wLight;						// interpolated world sp illum dir
in vec3 wLight2;					// interpolated world sp illum2 dir
in vec2 texcoord;					// Texture -> bmeinkr 55. dia

out vec4 fragmentColor;				// output goes to frame buffer

void main() {
    
    vec3 kd = texture(samplerUnit, texcoord).rgb;
	vec3 ks = vec3(1.0, 1.0, 1.0);
    float shine = 50.0;
    vec4 ka = vec4((kd * 0.35), 1);
    
    vec3 N = normalize(wNormal);
    vec3 V = normalize(wView);
    vec3 L = normalize(wLight);
    vec3 L2 = normalize(wLight2);
    
    vec3 H = normalize(L + V);
    vec3 H2 = normalize(L2 + V);
    
    float cosa = max(dot(N, L), 0.0f);
    float cosd = max(dot(N, H), 0.0f);
    
    float cosa2 = max(dot(N, L2), 0.0f);
    float cosd2 = max(dot(N, H2), 0.0f);
    
    vec3 result = ( vec3(kd * cosa) + vec3(ks * pow(cosd, shine)) ) * Le.rgb;
    vec3 result2 = ( vec3(kd * cosa2) + vec3(ks * pow(cosd2, shine)) ) * Le2.rgb;
    fragmentColor = La * ka + vec4(result, 0.5) + vec4(result2, 0.5);
    
}
)";


struct mat4 {
	float m[4][4];
public:
	mat4() {
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				if (i == j)
					m[i][j] = 1.0f;
				else
					m[i][j] = 0.0f;
			}
		}
	}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}

	mat4 operator*(const float c) {
		mat4 result;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				result.m[i][j] = m[i][j] * c;
			}
		}
		return result;
	}

	operator float*() { return &m[0][0]; }

	mat4 operator+(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				result.m[i][j] = m[i][j] + right.m[i][j];
			}
		}
		return result;
	}

	// bmeink 13. dia
	void SetUniform(unsigned shaderProg, char * name) {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniformMatrix4fv(loc, 1, GL_TRUE, &m[0][0]);
	}


};

struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	vec4 mul(const vec4& b) const {
		return vec4(v[0] * b.v[0], v[1] * b.v[1], v[2] * b.v[2]);
	}

	vec4 operator&(const float t[]) {
		return vec4(v[0] * t[0], v[1] * t[1], v[2] * t[2]);
	}

	vec4 operator+(const vec4& b) const {
		return vec4(v[0] + b.v[0], v[1] + b.v[1], v[2] + b.v[2]);
	}
	vec4 operator-(const vec4& b) const {
		return vec4(v[0] - b.v[0], v[1] - b.v[1], v[2] - b.v[2]);
	}
	float operator*(const vec4& b) const {
		return v[0] * b.v[0] + v[1] * b.v[1] + v[2] * b.v[2];
	}
	vec4 operator%(const vec4& b) const {
		return vec4(v[1] * b.v[2] - v[2] * b.v[1], v[2] * b.v[0] - v[0] * b.v[2], v[0] * b.v[1] - v[1] * b.v[0]);
	}
	vec4 operator*(const float c) const {
		return vec4(v[0] * c, v[1] * c, v[2] * c);
	}
	vec4 operator/(const float c) const {
		return vec4(v[0] / c, v[1] / c, v[2] / c);
	}
	friend vec4 operator*(float c, const vec4& v) {
		return v*c;
	}
	friend vec4 operator/(float c, const vec4& v) {
		return v / c;
	}
	vec4& operator+=(const vec4& b) {
		v[0] += b.v[0], v[1] += b.v[1], v[2] += b.v[2]; return *this;
	}
	vec4& operator-=(const vec4& b) {
		v[0] -= b.v[0], v[1] -= b.v[1], v[2] -= b.v[2]; return *this;
	}
	vec4& operator%=(const vec4& b) {
		v[0] = v[1] * b.v[2] - v[2] * b.v[1]; v[1] = v[2] * b.v[0] - v[0] * b.v[2]; v[2] = v[0] * b.v[1] - v[1] * b.v[0]; return *this;
	}
	vec4 operator-() const {
		return vec4(-v[0], -v[1], -v[2]);
	}
	float length() const {
		return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
	}
	vec4 normalize() const {
		float l = length(); if (l > 0.000003f) {
			return (*this / l);
		}
		else {
			return vec4(0, 0, 0, 1);
		}
	}

	// bmeinkr 13. dia
	void SetUniform(unsigned shaderProg, char * name) {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniform4f(loc, v[0], v[1], v[2], v[3]);
	}


};

//Transzformacios matrixok
mat4 Translate(vec4 t) {
	return mat4(1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		t.v[0], t.v[1], t.v[2], 1);
}

mat4 Scale(vec4 s) {
	return mat4(s.v[0], 0, 0, 0,
		0, s.v[1], 0, 0,
		0, 0, s.v[2], 0,
		0, 0, 0, 1);
}

//wikipedia ->rotation mtx
mat4 Rotate(float angle, vec4 w) {
	float cosa = cosf(angle);
	float sina = sinf(angle);
	mat4 wtensor = mat4(w.v[0] * w.v[0], w.v[1] * w.v[0], w.v[2] * w.v[0], 0,
		w.v[0] * w.v[1], w.v[1] * w.v[1], w.v[2] * w.v[1], 0,
		w.v[0] * w.v[2], w.v[1] * w.v[2], w.v[2] * w.v[2], 0,
		0, 0, 0, 1);

	mat4 wcross = mat4(0, -w.v[2], w.v[1], 0,
		w.v[2], 0, -w.v[0], 0,
		-w.v[1], w.v[0], 0, 0,
		0, 0, 0, 1);

	return mat4(1, 0, 0, 0,
				0, 1, 0, 0,
				0, 0, 1, 0,
				0, 0, 0, 1) * cosa + wcross * sina + wtensor * (1.0f - cosa);
}


// bmeinkr 6. dia
struct VertexData {
	vec4 position, normal;
	float u, v;
};

//bmeinkr 6. dia geometry + paramsurface
struct ParamSurface {
	unsigned int vao, nVtx;
	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N, int M) {

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		// bmeinkr 7. dia
		nVtx = N * M * 6;
		unsigned int vbo;
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		VertexData *vtxData = new VertexData[nVtx], *pVtx = vtxData;
		for (int i = 0; i < N; i++)
			for (int j = 0; j < M; j++) {
				*pVtx++ = GenVertexData((float)i / N, (float)j / M);
				*pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
				*pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
				*pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
				*pVtx++ = GenVertexData((float)(i + 1) / N, (float)(j + 1) / M);
				*pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
			}

		int stride = sizeof(VertexData), sVec3 = sizeof(vec4);
		glBufferData(GL_ARRAY_BUFFER, nVtx * stride, vtxData, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);  // AttribArray 0 = POSITION
		glEnableVertexAttribArray(1);  // AttribArray 1 = NORMAL
		glEnableVertexAttribArray(2);  // AttribArray 2 = UV
		glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, stride, (void*)0);
		glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, stride, (void*)sVec3);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(2 * sVec3));
	}

	void Draw() {
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, nVtx);
	}
	
};

//bmeink 22. dia
class Camera {
public:

	float fov, asp, fp, bp; 

	vec4 wEye; 
	vec4 wLookat; 
	vec4 wVup; 

	mat4 V_tmp;

	Camera() {}

	    Camera(vec4 eye0, vec4 target0, vec4 vup0 , float fov0 = 0, float asp0 = 0, float fp0 = 0, float bp0 = 0){
	        wEye = eye0; 
			wLookat = target0; 
			wVup = vup0;
	        fov = fov0; 
			asp = asp0; 
			fp = fp0; 
			bp = bp0;
	    }

	mat4 V(){
		vec4 w = (wEye - wLookat).normalize();	// back
		vec4 u = (wVup % w).normalize();		// right
		vec4 v = w % u;							// up 

		V_tmp = mat4(u.v[0], v.v[0], w.v[0], 0.0f,
			u.v[1], v.v[1], w.v[1], 0.0f,
			u.v[2], v.v[2], w.v[2], 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f);

		return  Translate(-wEye) * V_tmp;
	}

	mat4 P(){
		float sy = 1 / tan(fov / 2);

		return  mat4(sy / asp, 0.0f, 0.0f, 0.0f,
			0.0f, sy, 0.0f, 0.0f,
			0.0f, 0.0f, -(fp + bp) / (bp - fp), -1.0f,
			0.0f, 0.0f, -2 * fp*bp / (bp - fp), 0.0f);
	}

};

Camera camera(vec4(3.0f, 0.0f, 2.0f), vec4(3.0f, 0.0f, 0.0f), vec4(0.0f, 1.0f, 0.0f, 1.0f), 45.0f, 1.0f, 0.1f, 100.0f);


// struct Texture a diasoron, 54. dia
struct TextureMaterial {
	unsigned int textureId;

	TextureMaterial() {}

	 void Create(int i) {
		int width, height;
		vec4 *image = LoadImage(width, height, i);
		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_FLOAT, image);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


		int samplerUnit = GL_TEXTURE0;
	}

	vec4 * LoadImage(int & width, int & height, int & i) {
		width = TextureSize;
		height = TextureSize;
		vec4 szin = vec4();

		static vec4 imageData[TextureSize * TextureSize * 3];
		for (int x = 0; x < width; x++) {


			for (int y = 0; y < height; y++) {

				if (x % 16 <= 8 && y % 16 <= 8)
					if (i == 1) {
						szin = vec4(1.0f, 0.0f, 0.0f, 0.5f);
					}
					else
					{
						szin = vec4(1.0f, 0.0f, 1.0f, 0.5f);
					}
					
				else if (x % 16>8 && y % 16>8)

					if (i == 1) {
						szin = vec4(1.0f, 0.0f, 0.0f, 0.5f);
					}
					else
						{
						szin = vec4(1.0f, 0.0f, 0.0f, 0.5f);
					}
				else
					if (i == 1) {
						szin = vec4(1.0, 1.0f, 0, 0.5f);
					}
					else
					{
						szin = vec4(0.0f, 0.0f, 1.0f, 0.5f);
					}

				imageData[y * TextureSize + x] = szin;
			}
		}
		return imageData;
	}

	//levlistan felmerult kerdes alapjan javitva
	void Update() {
		int location = glGetUniformLocation(shaderProgram, "samplerUnit");
		glUniform1i(location, 0);
		glActiveTexture(0);
		glBindTexture(GL_TEXTURE_2D, textureId);
	}
};

struct Objects : TextureMaterial {
public:	
	vec4 ks = vec4(1.0, 1.0, 1.0, 1.0);
	float Shine = 50.0f;
	
	Objects() {}
	virtual void Animate(float time) { }

	void Update() {
		ks.SetUniform(shaderProgram, "ks");
	}
};


//diasor bmeanim 28. dia
struct Torus : public ParamSurface, public Objects {
public:
	float R, r;

	vec4 position, position_prev; float radius;
	float osc_u, osc_v, t_period;
	vec4 rotAxis;	float rotAng;

	Torus(float r0, float r1) {
		r = r0;
		R = r1;
	}

	VertexData GenVertexData(float u, float v) override {
		VertexData vd;
		float th = u * 2 * M_PI;
		float fi = v * 2 * M_PI;

		vd.position = vec4((R + r * cos(th)) * cos(fi),
			r * sin(th),
			(R + r * cos(th)) * sin(fi));

		vd.normal = -vec4(cos(fi) * r * cos(th),
			r * sin(th),
			sin(fi) * r * cos(th));

		vd.u = u;
		vd.v = v;

		return vd;
	}

	void Create() {
		TextureMaterial::Create(2);
		ParamSurface::Create(100, 100);
	}

	// diasor bmeincr - 23. dia
	void Draw() {
		mat4 M = Translate(position);
		mat4 Minv = Translate(-position);

		mat4 MVP = M * camera.V() * camera.P();

		MVP.SetUniform(shaderProgram, "MVP");
		Minv.SetUniform(shaderProgram, "Minv");
		M.SetUniform(shaderProgram, "M");

		TextureMaterial::Update();
		ParamSurface::Draw();
	}
};

Torus torus(1, 3);

struct Sphere : public ParamSurface,  public Objects {


	vec4 position, position_prev, lastPos;	
	float radius;							
	float t = 0;
	vec4 rotAxis;	float rotAng;			

	Sphere(vec4 c, float r){
		position = c;
		radius = r;
		rotAng = 0.0f; 
	}

	//bmeinkr 8. dia
	VertexData GenVertexData(float u, float v) override {
		VertexData vd;

		float fi = v * M_PI;
		float th = u * 2 * M_PI;

		vd.normal = vec4(cos(th) * sin(fi),
			sin(th) * sin(fi),
			cos(fi));

		vd.position = vd.normal * radius + position;
		vd.u = u; vd.v = v;

		return vd;
	}

	void Animate(float time) override {

		float R = torus.R;
		float r = torus.r;

		float period = 3.0f;

		t = fmodf(time, period);

		lastPos = position;

		rotAng += 0.1;


		float u = t / period * 2 * R * M_PI; 
		float v = t / period * 2 * r * M_PI;

		position = vec4((R + r * cos(u)) * cos(v), r * sin(u), (R + r * cos(u)) * sin(v));

	}

	void Create() {
		
		TextureMaterial::Create(1);
		ParamSurface::Create(100, 100);
	}

	// diasor bmeincr - 23. dia
	void Draw() {

		vec4 normal = ((((position.mul(vec4(1, 0, 1, 1))) - torus.position).normalize() * torus.R) - position).normalize();

		position = position + radius*normal;

		mat4 M = Rotate(rotAng, (normal % (position - lastPos)).normalize()) * Translate(position);
		mat4 Minv = Translate(-position) * Rotate(-rotAng, (normal % (position - lastPos).normalize()).normalize());

		mat4 MVP = M * camera.V() * camera.P();

		MVP.SetUniform(shaderProgram, "MVP");
		Minv.SetUniform(shaderProgram, "Minv");
		M.SetUniform(shaderProgram, "M");

		TextureMaterial::Update();
		ParamSurface::Draw();
	}
};

Sphere sphere(vec4(0, 0, 0), 0.3f);

struct LightBall {

	vec4 speed;

	vec4 dir;
	vec4 pos;

	vec4 color;
	int select;

	LightBall(vec4 p0, int select0, vec4 color0, vec4 speed0) {
		pos = p0;
		select = select0;
		color = color0;
		speed = speed0;
	}

	void Animate(float dt) {

		float R = torus.R;
		float r = torus.r;

		pos = pos + speed * dt; 

		//wikipedia -> torus formula
		float left = powf(R - sqrtf(pos.v[2] * pos.v[2] + pos.v[0] * pos.v[0]), 2.0f) + (pos.v[1] * pos.v[1]);


		if (left > (r*r)) {

			vec4 normal = ((((pos.mul(vec4(1, 0, 1, 1))) - vec4()).normalize() * R) - pos).normalize();

			//reflect
			speed = (speed - normal * (normal * speed) * 2.0f).normalize();

			while (left > (r*r)) {
				pos = pos + normal*0.1f;
				left = powf(R - sqrtf(pos.v[2] * pos.v[2] + pos.v[0] * pos.v[0]), 2.0f) + (pos.v[1] * pos.v[1]);
			}
		}

	}

	void Update() {
		if (select == 1) {
			pos.SetUniform(shaderProgram, "light_pos");
			color.SetUniform(shaderProgram, "Le");
		}

		if (select == 2) {
			pos.SetUniform(shaderProgram, "light_pos2");
			color.SetUniform(shaderProgram, "Le2");
		}
	}

};

LightBall ball1(vec4(3.0f, -0.5f, 0.0f), 1, vec4(0.0f, 0.3f, 0.3f), vec4(0.0f, 0.5f, 0.5f));
LightBall ball2(vec4(2.5f, 2.5f, 0.0f), 2, vec4(0.3f, 0.3f, 0.0f), vec4(-1.0f, 0.0f, 0.0f));


class Scene {
public:
	void Render() {

		
		vec4 La = vec4(0.35, 0.35, 0.35, 1.0);

		La.SetUniform(shaderProgram, "La");
		

		//create objects
		sphere.Create(); 
		torus.Create(); 

		sphere.Update();
		torus.Update();

	}

	void Animate(float sec, float dt) {

		sphere.Animate(sec);
		ball1.Animate(dt);
		ball2.Animate(dt);
	}

};

Scene scene;
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);


	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect Attrib Arrays to input variables of the vertex shader
	glBindAttribLocation(shaderProgram, 0, "vtxPos"); // vertexPosition gets values from Attrib Array 0
	glBindAttribLocation(shaderProgram, 1, "vtxNorm");    // vertexColor gets values from Attrib Array 1
														  //glBindAttribLocation(shaderProgram, 2, "uv");    // uv gets values from Attrib Array 2

														  // Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

																// program packaging
	glLinkProgram(shaderProgram);

	int OK;
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(shaderProgram);
	}

	// make this program run
	glUseProgram(shaderProgram);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glDisable(GL_CULL_FACE);

	scene.Render();
}

void onIdle() {

	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	float dt = (time - last_time) / 1000.0f;
	last_time = time;

	scene.Animate(sec, dt);

	glutPostRedisplay();
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen


	sphere.Draw();
	torus.Draw();

	ball1.Update();
	ball2.Update();

	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
	if (key == ESC) { glDeleteProgram(shaderProgram); exit(0); }
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(600, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
