#include "astrocv.h"

void square(double *data, int rows, int cols, double *out, int *rows_out, int *cols_out)
{
	int i, j, k;
	*cols_out = cols;
	*rows_out = cols;
	for (i = 0; i < cols; ++i)
		for (j = 0; j < cols; ++j)
		{
			out[i*cols + j] = 0.0;
			for (k = 0; k < rows; ++k)
				out[i*cols + j] += data[k*cols + i] * data[k*cols + j]; //JT[i,k] * J[k,j] = J[k,i] * J[k,j]
		}
}

void transpose(double *data, int rows, int cols, double *out, int *rows_out, int *cols_out)
{
	int i, j;
	*cols_out = rows;
	*rows_out = cols;
	for (i = 0; i < rows; i++)
		for (j = 0; j < cols; j++)
			out[j*rows + i] = data[i*cols + j];
}

void multBy(double *data, int rows, int cols, double *Other, int rows_oth, int cols_oth, double *out, int *rows_out, int *cols_out)
{
	int i, j, k;
	int nrows, ncols;
	double *o, *M;
	if (cols != rows_oth)
		return;
	*rows_out = rows;
	*cols_out = cols_oth;
	nrows = rows, ncols = cols_oth;
	o = Other;
	M = out;
	for (i = 0; i < nrows; i++)
		for (j = 0; j < ncols; j++)
		{
			M[i*ncols + j] = 0.0;
			for (k = 0; k < cols; k++)
				M[i*ncols + j] += data[i*cols + k] * o[k*ncols + j];
		}
}

static void gauss(double *mA, int mA_rows, int mA_cols, double *mB, int mB_rows, int mB_cols, double *out, int out_rows, int out_cols, int *Ok)
{
	int i, j, k;
	int N = mA_rows;
	double *A = mA;
	double *B = mB;
	double *X = out;
	int *columns = (int*)malloc(N * sizeof(*columns));

	if (Ok)
		*Ok = 0;

	if (mA_rows != mA_cols || mA_rows != mB_rows || mB_cols != 1)
		return;

	for (i = 0; i < N; ++i)
		columns[i] = i;

	for (i = 0; i < N; ++i)
	{ //diagonalization
		double a;
		int best_row = i, best_col = i;
		for (j = i + 1; j < N; ++j)
			for (k = i + 1; k < N; ++k)
				if (fabs(A[best_row*N + best_col]) < fabs(A[j*N + k]))
				{
					best_row = j;
					best_col = k;
				}

		if (best_row != i)
		{
			double tmp;
			//swap i-th and best_row-th rows
			for (j = i; j < N; ++j)
			{
				double tmp = A[i*N + j];
				A[i*N + j] = A[best_row*N + j];
				A[best_row*N + j] = tmp;
			}
			tmp = B[i];
			B[i] = B[best_row];
			B[best_row] = tmp;
		}

		if (best_col != i)
		{
			int tmp;
			//swap i-th and best_col-th cols
			for (j = 0; j < N; ++j)
			{
				double tmp = A[j*N + i];
				A[j*N + i] = A[j*N + best_col];
				A[j*N + best_col] = tmp;
			}
			tmp = columns[i];
			columns[i] = columns[best_col];
			columns[best_col] = tmp;
		}

		if (fabs(A[i*N + i]) < EPSILON)
			a = 0.0; // just ignore this column, solution element is zero
		else
			a = 1.0 / A[i*N + i];
		B[i] *= a;
		for (j = i; j < N; ++j)
			A[i*N + j] *= a;
		for (k = i + 1; k < N; ++k)
		{ //subtract i-th row from k-th row
			if (fabs(A[k*N + i]) < EPSILON)
				continue;
			a = 1.0 / A[k*N + i];
			B[k] *= a;
			for (j = i; j < N; ++j)
				A[k*N + j] *= a;
			B[k] -= B[i];
			for (j = i; j < N; ++j)
				A[k*N + j] -= A[i*N + j];
		}
	}
	for (i = N - 1; i >= 0; --i)
	{ //found solution
		int icol = columns[i];
		X[icol] = B[i];
		for (k = i - 1; k >= 0; --k)
			B[k] -= X[icol] * A[k*N + i];
	}

	if (Ok)
		*Ok = 1;

	free(columns);

	return;
}

double calcBiPolyValue(double *Coefficients, int count_Coeffs, double X, double Y)
{
	int i, j;
	double *temp = (double*)malloc(count_Coeffs * sizeof(double));
	int temp_size = 0;
	double ret = 0;
	int index = 0;
    
    temp[temp_size] = 1.0;
    temp_size++;
	
    for (i = 0; i < count_Coeffs; i++)
	{
		ret += Coefficients[i] * temp[index++];
		if (index >= temp_size)
		{
			temp[temp_size] = temp[temp_size - 1] * Y;
            temp_size++;
			for (j = 0; j < temp_size - 1; j++)
				temp[j] *= X;
			index = 0;
		}
	}
	free(temp);
	return ret;
}


double fitToBiPoly(const double *Xs, const double *Ys, const double *Zs, int count, double *Coefficients, int NCoefs, int *count_Coeffs)
{
	int i, j, k;
	//N - rows, M - cols
	int temp_size = 0;
	int ok = 0;
	int N = count;
	int M = MIN(N, NCoefs);

	double *J = (double*)malloc(N * M * sizeof(double));
	double *F = (double*)malloc(N * 1 * sizeof(double));
	double *pJ = J, *pF = F;
	double *temp = (double*)malloc(N * M * sizeof(double));

	double *A = (double*)malloc(M * M * sizeof(double));
	int A_rows = 0, A_cols = 0;
	
	double *B_tmp = (double*)malloc(N * M * sizeof(double));
	int Bt_rows = 0, Bt_cols = 0;

	double *B = (double*)malloc(count * 1 * sizeof(double));
	int B_rows = 0, B_cols = 0;

	double *X, ret = 0.0;

	for (i = 0; i < N; i++)
	{
		int index = 0;
		temp_size = 0;
		pF[i] = Zs[i];
		temp[temp_size++] = 1.0;
		for (j = 0; j < M; j++)
		{
			pJ[i*M + j] = temp[index++];
			if (index >= temp_size)
			{
				temp[temp_size] = temp[temp_size - 1] * Ys[i];
				temp_size++;
				for (k = 0; k < temp_size - 1; k++)
					temp[k] *= Xs[i];
				index = 0;
			}
		}
	}
	
	square(J, N, M, A, &A_rows, &A_cols);
	transpose(J, N, M, B_tmp, &Bt_rows, &Bt_cols);
	multBy(B_tmp, Bt_rows, Bt_cols, F, N, 1, B, &B_rows, &B_cols);

	X = (double*)malloc(A_cols * 1 * sizeof(double));
	gauss(A, A_rows, A_cols, B, B_rows, B_cols, X, A_cols, 1, &ok);
	
	if (!ok)
	{
		for (i = 0; i < NCoefs; i++)
			Coefficients[(*count_Coeffs)++] = 0.0;
		return 1e10;
	}
	for (i = 0; i < NCoefs; i++)
		Coefficients[(*count_Coeffs)++] = X[0 * 1 + i];// (0, i);


	for (i = 0; i < N; i++)
		ret += pow((Zs[i] - calcBiPolyValue(Coefficients, *count_Coeffs, Xs[i], Ys[i])), 2);////
	ret = sqrt(ret / (double)N);

	free(J);
	free(F);
	free(temp);
	free(A);
	free(B);
	free(B_tmp);
	free(X);
	return ret;
}
