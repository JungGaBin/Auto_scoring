#include "opencv2\opencv.hpp"
#include <iostream>
#include <cstring>
#include <vector>
#include <math.h>
#include <Windows.h>

using namespace cv;
using namespace std;

#define STRONG 255
#define WEAK 128
#define MAX_CORNER 5000
int harris_num = 0;

typedef struct _CornerPoints {
	int num;
	int x[MAX_CORNER];
	int y[MAX_CORNER];
} CornerPoints;
CornerPoints harris;

CornerPoints DibHarrisCorner(BYTE** in,BYTE** ptr,int H,int W) {
	int i, j, x, y;
	double threshold = 53000.0;
	int w = W;
	int h = H;
	ptr = in;
	double** dx2 = new double*[h];
	double** dy2 = new double*[h];
	double** dxy = new double*[h];

	for (int i = 0; i < h; i++) {
		dx2[i] = new double[w];
		dy2[i] = new double[w];
		dxy[i] = new double[w];
		memset(dx2[i], 0, sizeof(int)*w);
		memset(dy2[i], 0, sizeof(int)*w);
		memset(dxy[i], 0, sizeof(int)*w);
	}
	double tx, ty;
	for (j = 1; j < h - 1; j++) {
		for (i = 1; i < w - 1; i++) {
			tx=(ptr[j-1][i+1]+ptr[j][i+1]+ptr[j+1][i+1]
				- ptr[j - 1][i - 1] - ptr[j][i - 1] - ptr[j + 1][i - 1]) / 6.0;
			ty = (ptr[j + 1][i - 1] + ptr[j+1][i] + ptr[j + 1][i + 1]
				- ptr[j - 1][i - 1] - ptr[j-1][i] - ptr[j - 1][i + 1]) / 6.0;
			dx2[j][i] = tx*tx;
			dy2[j][i] = ty*ty;
			dxy[j][i] = tx*ty;

		}
	}
	double** gdx2 = new double*[h];
	double** gdy2 = new double*[h];
	double** gdxy = new double*[h];
	for (int i = 0; i < h; i++) {
		gdx2[i] = new double[w];
		gdy2[i] = new double[w];
		gdxy[i] = new double[w];
		memset(gdx2[i], 0, sizeof(double)*w);
		memset(gdy2[i], 0, sizeof(double)*w);
		memset(gdxy[i], 0, sizeof(double)*w);
	}
	double g[5][5] = { {1,4,6,4,1},{4,16,24,16,24},
	{6,24,36,24,6},{4,16,24,16,24},{1,4,6,4,1} };
	for (y = 0; y < 5; y++) {
		for (x = 0; x < 5; x++) {
			g[y][x] /= 256.;
		}
	}
	double tx2, ty2, txy;
	for (j = 2; j < h - 2; j++) {
		for (i = 2; i < w - 2; i++) {
			tx2 = ty2 = txy = 0;
			for (y = 0; y < 5; y++) {
				for (x = 0; x < 5; x++) {
					tx2 += (dx2[j + y - 2][i + x - 2] * g[y][x]);
					ty2 += (dy2[j + y - 2][i + x - 2] * g[y][x]);
					txy += (dxy[j + y - 2][i + x - 2] * g[y][x]);
				}
			}
			gdx2[j][i] = tx2;
			gdy2[j][i] = ty2;
			gdxy[j][i] = txy;
		}
	}
	double** crf = new double*[h];
	for (i = 0; i < h; i++) {
		crf[i] = new double[w];
		memset(crf[i], 0, sizeof(double)*w);
	}
	double k = 0.04;
	for (j = 2; j < h - 2; j++) {
		for (i = 2; i < w - 2; i++) {
			crf[j][i] = (gdx2[j][i] * gdy2[j][i] - gdxy[j][i] * gdxy[j][i])
				- k*(gdx2[j][i] + gdy2[j][i])*(gdx2[j][i] + gdy2[j][i]);
		}
	}
	CornerPoints cp;
	cp.num = 0;
	for (j = 2; j < h - 2; j++) {
		for(i=2;i<w-2;i++){
			if (crf[j][i] > threshold) {
				if (crf[j][i] > crf[j - 1][i] && crf[j][i] > crf[j - 1][i + 1] &&
					crf[j][i] > crf[j][i + 1] && crf[j][i] > crf[j + 1][i + 1] &&
					crf[j][i] > crf[j + 1][i] && crf[j][i] > crf[j + 1][i - 1] &&
					crf[j][i] > crf[j][i - 1] && crf[j][i] > crf[j - 1][i - 1]) {
					if (cp.num < MAX_CORNER) {
						cp.x[cp.num] = i;
						cp.y[cp.num] = j;
						cp.num++;
					}
				}
			}
		}
	}
	for (int i = 0; i < h; i++) {
		delete [] dx2[i];
		delete [] dy2[i];
		delete [] dxy[i];
		delete [] gdx2[i];
		delete [] gdy2[i];
		delete [] gdxy[i];
		delete [] crf[i];
	}
	delete[] dx2;
	delete[] dy2;
	delete[] dxy;
	delete[] gdx2;
	delete[] gdy2;
	delete[] gdxy;
	delete[] crf;
	
	harris_num = cp.num;
	harris = cp;
	return cp;
}
typedef unsigned char BYTE;
/*typedef struct _circle {
	int h;
	int w;
	int r;

	struct _circle(int h = 0, int w = 0, int r = 0)
		: h(h), w(w), r(r) {}
}Circle;*/
class Circle {
public:
	int h;
	int w;
	int r;
	int vote;

	Circle(int h = 0, int w = 0, int r = 0, int vote = 0)
		:h(h), w(w), r(r), vote(vote) {}
};

double score = 0.0;

vector<Circle> circles;
vector<Circle> result;
typedef unsigned char BYTE;

bool h_compare(const Circle lhs, const Circle& rhs)
{
	return lhs.h < rhs.h;
}

bool w_compare(const Circle lhs, const Circle& rhs)
{
	return lhs.w < rhs.w;
}

bool vote_compare(const Circle lhs, const Circle& rhs)
{
	return lhs.vote < rhs.vote;
}
double** MemAllocDouble(int H, int W, double init);
void MemFreeDouble(double** mem, int H);
BYTE** MemAllocBYTE(int H, int W, BYTE init);
void MemFreeBYTE(BYTE** mem, int H);

void CannyEdge(BYTE** in, BYTE**& out, int H, int W, double sigma, double low, double high);
void GaussianFilter(BYTE** in, double**& out, int H, int W, double sigma);
void CircleHoughTransfrom(BYTE** in, BYTE**& out, int init_h,int init_w,int H, int W, int rMin, int rMax, int threshold,int problem);
void RemoveOverlappedCircle();
void DrawCircleAndDisplay(BYTE** out, int H, int  W);
void DrawCircle(BYTE** out, int H, int W,int popup);
void LineHoughTransform(BYTE** in, BYTE**& out, int init_h, int init_w, int H, int W, int thres, int problem);
BYTE** copyToArray(Mat& input);

int check = 0;
int problem = 1;
int overlap = 0;
int solution[10];
int main()
{
	int H, W;
	//영상 읽어오기
	//Mat input = imread("O_overlap.jpg", 0);
	//test10 : X가 겹칠때
	//test12 : 정상이미지
	//XO_overlap : XO겹칠때

	Mat input = imread("test10.jpg", 0);
	H = input.rows, W = input.cols;
	BYTE** in = copyToArray(input);
	BYTE** canny;
	//BYTE** label;
	double sigma, low, high;
	/*
	sigma값이랑 low threshold, high threshold 값 지정

	cout << "input sigma, low, high...." << endl;
	cin >> sigma >> low >> high;
	*/

	printf("세로 %d 가로 %d \n", H, W);
	if (W % 2 == 0)
		W = W;
	else
		W = W - 1;

	printf("세로 %d 가로 %d \n", H, W);
	//일단은 아랫 값으로 테스트
	sigma = 0.7, low = 45, high = 90;
	CannyEdge(in, canny, H, W, sigma, low, high);
	DibHarrisCorner(in, canny, H, W);

	printf("harrisnum %d \n", harris_num);
	for (int i = 0; i < harris_num; i++) {
		if(i==0)
			printf("harrisnum :%d %d %d  \n", i, harris.x[i], harris.y[i]);
		else
			printf("harrisnum :%d %d %d  %d \n", i, harris.x[i], harris.y[i],harris.y[i]-harris.y[i-1]);
	}

	printf("harrisnum %d \n", harris_num);
	for (int i = 0; i < harris_num; i++) {
		for(int j=0;j<10;j++)
			canny[harris.y[i]+j][harris.x[i]+j] = 255;
	}
	for (int i = 0; i < harris_num; i++) {
		for (int j = 0; j<10; j++)
			canny[harris.y[i] - j][harris.x[i] - j] = 255;
	}
	
	//출력하기
	Mat output(H, W, CV_8UC1);
	for (int h = 0; h < H; ++h) {
		for (int w = 0; w < W; ++w) {
			output.at<BYTE>(h, w) = canny[h][w];
		}
	}
	imwrite("canny.bmp", output);
	imshow("Canny Edge", output);
	waitKey(0);
	int midpoint = W / 2;
	int problem_h[10] = { 0, };
	int c;
	int d1 = 0;
	int R_pixel = 0;
	int R_sum = 0;
	// 구분선 긋기
	for(int h=0;h<H;h++){
		c = 0;
		R_pixel = 0;
		for (int w = 0; w < midpoint; w++) {
			if (canny[h][w] == 0) // 그대로 n번째 문제 유지
				c++;
			else{
				R_pixel++;
			}
		}
		// c가 midpoint값과 같다면? 모든 픽셀이 0이라고 알 수 있다.
		if (c != midpoint) {
			problem = problem;
			d1 = 1;
			//printf("%d,",R_pixel);
			R_sum += R_pixel;
			//printf("[ %d ], %d %d \n", problem,h,R_pixel);
			
		}
		else {
			if (d1 == 1) {
				printf("다음 문제로 This problem is %d ,%d\n", problem,h);
				problem = problem + 1;
				problem_h[problem-2] = h;
				d1 = 0;
				R_sum = 0;
				
			}
			else if(d1==0){
				//printf("아직 아님 This problem is %d , %d \n", problem, h);
				problem = problem;
			}			
			//그런데 만약에 픽셀값이 모두 0이고, 픽셀값이 들어왔다가 다시 0이된경우 problem이 +되어야 한다.
		}

	}
	int left_problem = problem - 1;
	printf("반쪽 %d\n", left_problem);
	d1 = 0;
	for (int h = 0; h<H; h++) {
		c = 0;
		R_pixel = 0;
		for (int w = midpoint; w < W; w++) {
			if (canny[h][w] == 0)
				c++;
			else
				R_pixel++;
		}
		// c가 midpoint-1값과 같다면? 모든 픽셀이 0이라고 알 수 있다.
		if (c == midpoint) {
			if (d1 == 1) {
				printf("다음 문제로 This problem is %d , %d \n", problem, h);
				problem = problem + 1;
				problem_h[problem-2] = h;
				d1 = 0;
				if (problem > 10) {
					break;
				}
			}
			else if (d1 == 0) {
				//printf("아직 아님 This problem is %d , %d \n", problem, h);
				problem = problem;
			}
			//그런데 만약에 픽셀값이 모두 0이고, 픽셀값이 들어왔다가 다시 0이된경우 problem이 +되어야 한다.
		}
		else {//다르다면 픽셀값이 들어왔다고 가정한다.
			problem = problem;
			d1 = 1;
			//printf("[ %d ], %d \n", problem, R_pixel);
		}
	}
	BYTE** out;

	for (int i = 0; i < 10; i++) {
		if (problem_h[i] == 0)
			overlap = 1;
	}
	for (int i = 0; i < 10; i++){
		if (overlap == 1) printf("겹치는 부분이 있음");
		printf("%d %d\n", i, problem_h[i]);
	}

	if(overlap==0){
		for (int j = 0; j<left_problem; j++) {
			if (j == 0) {
				CircleHoughTransfrom(canny, out, 0, 0, problem_h[0], W / 2, 15, 50, 135,j);
				DrawCircle(in, H, W,j);
			}
			else {
				CircleHoughTransfrom(canny, out, problem_h[j - 1], 0, problem_h[j], W / 2, 15, 50, 135, j);
				DrawCircle(in, H, W,j);
			}
		}
		for (int j = left_problem; j<10; j++) {
			if (j == left_problem) {
				CircleHoughTransfrom(canny, out, 0, W / 2, problem_h[j], W, 15, 50, 135, j);
				DrawCircle(in, H, W,j);
			}
			else {
				CircleHoughTransfrom(canny, out, problem_h[j - 1], W / 2, problem_h[j], W, 15, 50, 135, j);
				DrawCircle(in, H, W,j);
			}
		}
		//DrawCircle(in, H, W);
	}
	else { // 겹치는 부분이 있음.

		printf("겹치는 코드 고쳐주세요 ");
		/*for(int i=0;i<10;i++){
			for (int h = 0; h < problem_h[i]; h++) {
				for (int w = 0; w < W / 2; w++) {
				}
			}
		}
		DibHarrisCorner(canny, out, problem_h[0], W / 2);
		printf("KIIKI : %d \n", harris.num);

		for (int j = 0; j<left_problem; j++) {
			if (j == 0) {
				CircleHoughTransfrom(canny, out, 0, 0, problem_h[0], W / 2, 15, 50, 130, j);
				DrawCircle(in, H, W,j);
			}
			else {
				CircleHoughTransfrom(canny, out, problem_h[j - 1], 0, problem_h[j], W / 2, 15, 50, 130, j);
				DrawCircle(in, H, W,j);
			}
		}
		for (int j = left_problem; j<10; j++) {
			if (j == left_problem) {
				CircleHoughTransfrom(canny, out, 0, W / 2, problem_h[j], W, 15, 50, 130, j);
				DrawCircle(in, H, W,j);
			}
			else {
				CircleHoughTransfrom(canny, out, problem_h[j - 1], W / 2, problem_h[j], W, 15, 50, 130, j);
				DrawCircle(in, H, W,j);
			}
			
		}*/
		
	}
	for (int i = 0; i < 10; i++)
		printf("[%d] %d\n", i + 1, solution[i]);

	int score = 0;
	if (solution[0] == 0) //o
		score += 10;
	if (solution[1] == 0) //o
		score += 10;
	if (solution[2] == 0) //o
		score += 10;
	if (solution[3] == 0) //o
		score += 10;
	if (solution[4] == 0) //o
		score += 10;
	if (solution[5] == 1) //o
		score += 10;
	if (solution[6] == 1) //o
		score += 10;
	if (solution[7] == 1) //o
		score += 10;
	if (solution[8] == 1) //o
		score += 10;
	if (solution[9] == 1) //o
		score += 10;

	printf("That is score %d \n", score);
	return 0;
}

double** MemAllocDouble(int H, int W, double init)
{
	double** rtn = new double*[H];
	for (int h = 0; h < H; ++h) {
		rtn[h] = new double[W];
		memset(rtn[h], init, sizeof(double) * W);
	}
	return rtn;
}

void MemFreeDouble(double** mem, int H)
{
	for (int h = 0; h < H; ++h)
		delete[] mem[h];
	delete[] mem;
}

BYTE** MemAllocBYTE(int H, int W, BYTE init)
{
	BYTE** rtn = new BYTE*[H];
	for (int h = 0; h < H; ++h) {
		rtn[h] = new BYTE[W];
		memset(rtn[h], init, sizeof(BYTE) * W);
	}
	return rtn;
}

void MemFreeBYTE(BYTE** mem, int H)
{
	for (int h = 0; h < H; ++h)
		delete[] mem[h];
	delete[] mem;
}

/*
<copyToArray>

Mat객체를 array형태로 바꾸는 함수
*/
BYTE** copyToArray(Mat& input)
{
	int W = input.cols;
	int H = input.rows;
	BYTE* pInput = input.data;

	BYTE** rtn = new BYTE*[H];
	for (int h = 0; h < H; ++h) {
		rtn[h] = new BYTE[W];
		for (int w = 0; w < W; ++w) {
			rtn[h][w] = pInput[h * W + w];
		}
	}
	return rtn;
}

/*
<GaussianFilter>

in: input 영상
out: 가우시안 필터 적용된 output -> CannyEdge에서 활용
H, W : 생략
sigma : 표준편차 값
*/
void GaussianFilter(BYTE** in, double**& out, int H, int W, double sigma)
{
	int masksize = (int)(2 * 4 * sigma + 1.0);
	if (masksize % 2 == 0) masksize++;
	int padSize = masksize / 2;

	out = MemAllocDouble(H, W, 0.0);

	double* mask = new double[masksize];
	for (int i = 0; i < masksize; ++i) {
		int x = i - padSize;
		mask[i] = exp(-(x * x) / (2 * sigma * sigma)) / (sqrt(2 * CV_PI) * sigma);
	}

	double** temp = MemAllocDouble(H, W, 0.0);

	//세로방향
	double sum1, sum2;
	for (int w = 0; w < W; ++w) {
		for (int h = 0; h < H; ++h) {
			sum1 = sum2 = 0.0;

			for (int m = 0; m < masksize; ++m) {
				int x = m - padSize + h;
				if (x >= 0 && x < H)
				{
					sum1 += mask[m];
					sum2 += (mask[m] * in[x][w]);
				}
			}
			temp[h][w] = sum2 / sum1;
		}
	}

	//가로방향
	for (int h = 0; h < H; ++h)
	{
		for (int w = 0; w < W; ++w)
		{
			sum1 = sum2 = 0.0;
			for (int m = 0; m < masksize; ++m)
			{
				int x = m - padSize + w;
				if (x >= 0 && x < W) {
					sum1 += mask[m];
					sum2 += (mask[m] * temp[h][x]);
				}
			}
			out[h][w] = sum2 / sum1;
		}
	}

	//메모리 해제
	delete[] mask;
	MemFreeDouble(temp, H);
}

/*
<CannyEdge>

in : input 영상
canny : output
H: 이미지 Height
W: 이미지 Width
low: Canny Edge 추출할 때 최소임계값
high: Canny Edge 추출할 때 최대임계값
:: low, high를 활용해 최종 경계선을 확정한다. (Hysteresis Edge Tracking)
*/
void CannyEdge(BYTE** in, BYTE**& out, int H, int W, double sigma, double low, double high)
{
	int h, w, m, n;
	double** gss;
	/*
	1. Gaussian Filter을 통과시켜서 Blur된 영상 만드는 과정
	*/
	GaussianFilter(in, gss, H, W, sigma);

	/*
	gW : 경계선 width 방향의 Gradient
	gH : 경계선 height 방향의 Gradient
	magin : 경계선의 세기값
	*/
	double** gW = MemAllocDouble(H, W, 0.0);
	double** gH = MemAllocDouble(H, W, 0.0);
	double** magni = MemAllocDouble(H, W, 0.0);

	/*
	2. 소벨마스크를 이용해 width/height 방향의 Gradient 계산
	+ 세기값도 계산
	*/
	for (h = 1; h < H - 1; ++h) {
		for (w = 1; w < W - 1; ++w)
		{
			gW[h][w] = -gss[h - 1][w - 1] - 2 * gss[h][w - 1] - gss[h + 1][w - 1]
				+ gss[h - 1][w + 1] + 2 * gss[h][w + 1] + gss[h + 1][w + 1];
			gH[h][w] = -gss[h - 1][w - 1] - 2 * gss[h - 1][w] - gss[h - 1][w + 1]
				+ gss[h + 1][w - 1] + 2 * gss[h + 1][w] + gss[h + 1][w + 1];

			magni[h][w] = sqrt(gW[h][w] * gW[h][w] + gH[h][w] * gH[h][w]);
		}
	}


	out = MemAllocBYTE(H, W, 0);
	vector<Point> strongEdges;

	int area;
	double theta;
	bool localMax;
	
	//요부분 부터는 이해못해서 나중에 다시 주석달게요
	//일단은 3. Non-maximum suppression(비최대 억제) 부분입니다.
	for (h = 1; h < H - 1; ++h) {
		for (w = 1; w < W - 1; ++w)
		{
			if (magni[h][w] > low)
			{
				if (gW[h][w] != 0.0)
				{
					theta = atan2(gH[h][w], gW[h][w]) * 180 / CV_PI;
					if (((theta >= -22.5f) && (theta < 22.5f)) || (theta >= 157.5f) || (theta < -157.5f))
						area = 0;
					else if (((theta >= 22.5f) && (theta < 67.5f)) || ((theta >= -157.5f) && (theta < -112.5f)))
						area = 45;
					else if (((theta >= 67.5) && (theta < 112.5)) || ((theta >= -112.5) && (theta < -67.5)))
						area = 90;
					else
						area = 135;
				}
				else
					area = 90;

				localMax = false;
				switch (area)
				{
				case 0:
					if ((magni[h][w] >= magni[h][w - 1]) && (magni[h][w] > magni[h][w + 1]))
						localMax = true;
					break;
				case 45:
					if ((magni[h][w] >= magni[h - 1][w - 1]) && (magni[h][w] > magni[h + 1][w + 1]))
						localMax = true;
					break;
				case 90:
					if ((magni[h][w] >= magni[h - 1][w]) && (magni[h][w] > magni[h + 1][w]))
						localMax = true;
					break;
				case 135:
				default:
					if ((magni[h][w] >= magni[h - 1][w + 1]) && (magni[h][w] > magni[h + 1][w - 1]))
						localMax = true;
					break;
				}

				if (localMax)
				{
					if (magni[h][w] > high) {
						out[h][w] = STRONG;
						strongEdges.push_back(Point(h, w));
					}
					else
						out[h][w] = WEAK;
				}
			}
		}
	}
	
	//4 .히스테리시스 엣지 트래킹
	while (!strongEdges.empty())
	{
		Point p = strongEdges.back();
		strongEdges.pop_back();

		int hTemp = p.x, wTemp = p.y;

		for (h = -1; h <= 1; ++h) {
			for (w = -1; w <= 1; ++w) {
				if (h == 0 && w == 0) continue;
				if (out[hTemp + h][wTemp + w] == WEAK) {
					out[hTemp + h][wTemp + w] = STRONG;
					strongEdges.push_back(Point(hTemp + h, wTemp + w));
				}
			}
		}
	}

	for (h = 0; h < H; ++h) {
		for (w = 0; w < W; ++w) {
			if (out[h][w] == WEAK)
				out[h][w] = 0;
		}
	}
	/*out[145][166] = 255;
	out[176][64] = 255;
	out[292][61] = 255;
	out[310][82] = 255;
	out[319][171] = 255;
	out[354][64] = 255;
	out[354][87] = 255;
	out[446][43] = 255;*/

	//메모리해제
	MemFreeDouble(gW, H);
	MemFreeDouble(gH, H);
	MemFreeDouble(magni, H);
}
void initAccum(int**& acc, int H, int W)
{
	acc = new int*[H];
	for (int h = 0; h < H; ++h) {
		acc[h] = new int[W];
		for (int w = 0; w < W; ++w) {
			acc[h][w] = 0;
		}
	}
}

void initLookupTable(double* LUT_COS, double* LUT_SIN)
{
	for (int angle = 0; angle < 360; ++angle) {
		LUT_SIN[angle] = sin(angle * CV_PI / 180.0);
		LUT_COS[angle] = cos(angle * CV_PI / 180.0);
	}
}
int result_circle = 0;
//int score[10];
void CircleHoughTransfrom(BYTE** in, BYTE**& out, int init_h,int init_w,int H, int W, int rMin, int rMax, int threshold,int problem)
{
	if (problem <=10){
	double lutSin[361];
	double lutCos[361];
	//Init Lookup Table
	initLookupTable(lutCos, lutSin);

	//int length = (int)(H / 2);
	int h, w, ang, a, b;
	int** accumulator;

	initAccum(accumulator, H, W);
	for (int r = rMin; r <= rMax; r++) {

		//Calc Accumulator Start
		for (h = init_h; h < H; ++h) {
			for (w = init_w; w < W; ++w)
			{
				if (in[h][w] > 0)
				{
					for (ang = 0; ang <= 360; ++ang)
					{
						a = h - round(r * lutSin[ang]);
						b = w - round(r * lutCos[ang]);
						if (a >= 0 && a < H && b >= 0 && b < W)
							accumulator[a][b] += 1;
					}
				}
			}
		}
		//End

		//Find max accum
		int accMax = -1;
		for (h = init_h; h < H; ++h)
			for (w = init_w; w < W; ++w) {
				if (accMax < accumulator[h][w]) {
					accMax = accumulator[h][w];
				}
			}
		//printf("radius : %d, max acc: %d\n", r, accMax);
		//End

		//Get Circle
		if (accMax >= threshold)
		{
			for (h = init_h; h < H; ++h) {
				for (w = init_w; w < W; ++w) {
					if (accumulator[h][w] < threshold)
						accumulator[h][w] = 0;
				}
			}

			for (h = init_h; h < H; ++h)
			{
				for (w = init_w; w < W; ++w)
				{
					//Threshold 값을 넘는 경우 주변 픽셀의 accum값을 합산
					if (h > 0 && h < H - 1 && w > 0 && w < W - 1 && accumulator[h][w] >= threshold) {
						int sum = 0;
						for (int i = -1; i <= 1; ++i) {
							for (int j = -1; j <= 1; ++j) {
								sum += accumulator[h + i][w + j];
							}
						}

						double avg = (double)sum / 9.0;
						if (avg >= threshold / 9) {
							//printf("h : %d, w : %d, radius : %d, average : %lf\n", h, w, r, avg);
							circles.push_back(Circle(h, w, r, accumulator[h][w]));

							for (int i = h; i < h + 9; ++i)
								for (int j = w; j < w + 9; ++j)
									accumulator[i][j] = 0;
						}
					}
				}
			}
		}
		//End
		initAccum(accumulator, H, W);
	}//r loop end

	 //result.clear();
	 //result.assign(circles.begin(), circles.end());



	RemoveOverlappedCircle(); 
	int last_circle;
	last_circle = result_circle;
	result_circle = result.size();
	if (last_circle != result_circle) {
		if (last_circle + 1 == result_circle) {
			solution[problem] = 0;
			printf("[%d] this is o ,not overlap %d %d \n",problem, last_circle, result.size());
		}
		else {
			//for (int i = 0; i<result_circle-last_circle;i++)
				printf("[%d] this is o ,and overlap %d %d \n",problem, last_circle, result.size());
		}
	}
	else {
		solution[problem] = 1;
		printf("[%d] this is x %d %d\n", problem,last_circle, result.size());
		
	}
	
	}
	
	//printf("Complete Circle Hough Transform...\n");
}


//겹쳐지는 원 제거하는 함수
void RemoveOverlappedCircle()
{
	//일단 원의 원점(h, w)에서 h를 기준으로 정렬
	sort(circles.begin(), circles.end(), h_compare);
	vector<vector<Circle>> vvc;
	vector<Circle> vc;

	int size = circles.size();
	Circle c = circles[0];
	for (int i = 1; i < size; ++i)
	{
		int h = circles[i].h;
		int w = circles[i].h;

		//비슷한 h값을 가지는 원 원점들을 선택
		if ((c.h - c.r) <= h && h <= (c.h + c.r)) {
			vc.push_back(circles[i]);
		}
		else //그렇지 않을 경우 여태까지 선택한 원을 vvc 벡터에 저장 
		{
			vvc.push_back(vc);
			vc.clear();

			//새로운 원을 기준으로 선택
			c = circles[i];
			vc.push_back(c);
		}
	}

	vvc.push_back(vc);
	vc.clear();

	//vcc 백터의 한 행은 비슷한 h값을 가진 원들로 이루어져 있다.
	//각 행들을 w 값을 기준으로 정렬한다.
	size = vvc.size();
	for (int i = 0; i < size; ++i) {
		sort(vvc[i].begin(), vvc[i].end(), w_compare);

		int len = vvc[i].size();
		c = vvc[i][0];
		vc.push_back(c);
		for (int j = 1; j < len; ++j) {
			int h = vvc[i][j].h;
			int w = vvc[i][j].w;

			//비슷한 w값을 가진 원들을 vc벡터에 저장
			if ((c.w - c.r) <= w && w <= (c.w + c.r)) {
				vc.push_back(vvc[i][j]);
			}
			else
			{
				//vc에 저장된 원들 중에서 vote 값이 가장 큰 원을 선택한다.
				sort(vc.begin(), vc.end(), vote_compare);
				result.push_back(vc[vc.size() - 1]);
				vc.clear();

				//새로운 원을 기준으로 선택
				c = vvc[i][j];
				vc.push_back(c);
			}
		}
		sort(vc.begin(), vc.end(), vote_compare);
		result.push_back(vc[vc.size() - 1]);
		vc.clear();
	}
}

void LineHoughTransform(BYTE** in, BYTE**& out, int init_h, int init_w, int H, int W, int thres, int problem)
{
	register int i, j, h, w, k;
	int index, d;
	int vote[360][600] = { 0 };

	double LUT_SIN[361];
	double LUT_COS[361];

	out = MemAllocBYTE(H, W, 0);
	for (h = init_h; h < H; ++h) {
		for (w = init_w; w < W; ++w) {
			out[h][w] = in[h][w];
		}
	}
	float p2d = 3.141592654f / 180.0f;
	//Init Lookup Table
	for (i = 0; i<360; i++)
	{
		LUT_COS[i] = (float)cos(i*p2d);
		LUT_SIN[i] = (float)sin(i*p2d);
	}

	//For voting
	for (h = init_h; h < H; h++)
	{
		for (w = init_w; w<W; w++)
		{
			if (in[h][w] == 255)
			{
				for (k = 0; k<360; k++)
				{
					d = (int)(h * LUT_COS[k] + w * LUT_SIN[k]);
					if (d >= 4 && d <= 600) vote[k][d]++;
				}
			}
		}
	}
	int height = H, width = W;
	//For display
	for (d = 4; d <= 600; d++)
	{
		for (k = 0; k<360; k++)
		{
			if (vote[k][d] > thres)
			{
				printf("vote: %d, angle: %d, distance: %d\n", vote[k][d], k, d);
				i = j = 2;
				for (j = init_w+2; j<width; j++) // vertical pixel
				{
					//x = (distance - y * sin(theta)) / cos(theta)
					i = (int)((d - j*LUT_SIN[k]) / LUT_COS[k]);
					if (i<height && i>0) out[i][j] = 255;
				}
				for (i = init_h+2; i<height; i++) // horizontal pixel
				{
					//y = (distance - x * cos(theta)) / sin(theta)
					j = (int)((d - i*LUT_COS[k]) / LUT_SIN[k]);
					if (j<height && j>0) out[i][j] = 255;
				}
			}
		}
	}

	printf("Complete Line Hough Transform...\n");
	
}

void DrawCircleAndDisplay(BYTE** out, int H, int  W)
{
	int h, w;
	//Draw Circle
	Mat output(H, W, CV_8UC1);
	for (h = 0; h < H; ++h) {
		for (w = 0; w < W; ++w) {
			output.at<BYTE>(h, w) = out[h][w];
		}
	}
	Mat temp;
	cvtColor(output, temp, CV_GRAY2RGB);
	while (!circles.empty())
	{
		Circle c = circles.back();
		circles.pop_back();

		//printf("h : %d, w : %d, r: %d\n", c.h, c.w, c.r);
		circle(temp, Point(c.w, c.h), c.r, Scalar(0, 0, 255), 1);
	}

	imshow("Result", temp);
	imwrite("LineDetection.bmp", temp);
	waitKey(0);
}

void DrawCircle(BYTE** out, int H, int W,int popup)
{
	int h, w;
	//Draw Circle
	Mat output(H, W, CV_8UC1);
	for (h = 0; h < H; ++h) {
		for (w = 0; w < W; ++w) {
			output.at<BYTE>(h, w) = out[h][w];
		}
	}
	Mat temp;
	cvtColor(output, temp, CV_GRAY2RGB);
	while (!result.empty())
	{
		Circle c = result.back();
		result.pop_back();

		//printf("h : %d, w : %d, r: %d\n", c.h, c.w, c.r);
		circle(temp, Point(c.w, c.h), c.r, Scalar(0, 0, 255), 1);
	}

	imwrite("CircleDectection.bmp", temp);
	if(popup==9)
		imshow("Circle Detection", temp);
	waitKey(0);
}