#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAX_COLS 80
#define MAX_ROWS 1460
#define FILENAME "/house/house-p/house-p/data/train.csv"
#define OUTPUT_FILE "/house/house-p/house-p/data/processed_train.csv"  // 输出文件名

// 函数：读取CSV文件并将数据存入矩阵
void read_csv(const char *filename, double data[MAX_ROWS][MAX_COLS], int *num_rows, int *num_cols) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("无法打开文件 %s 进行读取。\n", filename);
        exit(1);
    }

    char line[1024];
    *num_rows = 0;
    *num_cols = 0;

    while (fgets(line, sizeof(line), file)) {
        char *token = strtok(line, ",");
        int col = 0;

        while (token) {
            data[*num_rows][col] = atof(token); // 将字符串转换为浮动数
            token = strtok(NULL, ",");
            col++;
        }

        (*num_rows)++;
        *num_cols = col;  // 假设每一行的列数相同
    }

    fclose(file);
}

// 并行处理缺失值（假设缺失值为-1）
void fillna_zero(double data[MAX_ROWS][MAX_COLS], int rows, int cols, int col_index) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        if (data[i][col_index] == -1) {  // 假设-1表示缺失值
            data[i][col_index] = 0;  // 填充为0
        }
    }
}

// 并行进行特征工程（例如创建新特征：TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF）
void create_features(double data[MAX_ROWS][MAX_COLS], int rows) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        data[i][MAX_COLS - 1] = data[i][5] + data[i][6] + data[i][7]; // 示例：TotalSF = 1stFlrSF + 2ndFlrSF + TotalBsmtSF
    }
}

// 串行处理缺失值（填充为0）
void fillna_zero_serial(double data[MAX_ROWS][MAX_COLS], int rows, int cols, int col_index) {
    for (int i = 0; i < rows; i++) {
        if (data[i][col_index] == -1) {
            data[i][col_index] = 0;
        }
    }
}

// 串行进行特征工程
void create_features_serial(double data[MAX_ROWS][MAX_COLS], int rows) {
    for (int i = 0; i < rows; i++) {
        data[i][MAX_COLS - 1] = data[i][5] + data[i][6] + data[i][7]; // 示例：TotalSF
    }
}

// 函数：将处理后的数据保存到CSV文件
void save_to_csv(const char *filename, double data[MAX_ROWS][MAX_COLS], int num_rows, int num_cols) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("无法打开文件 %s 进行写入。\n", filename);
        exit(1);
    }

    // 写入数据行
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            fprintf(file, "%.2f", data[i][j]);
            if (j < num_cols - 1) {
                fprintf(file, ",");
            }
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

int main() {
    double data[MAX_ROWS][MAX_COLS];
    int num_rows, num_cols;

    // 步骤1：读取CSV数据
    read_csv(FILENAME, data, &num_rows, &num_cols);
    printf("已加载 %d 行和 %d 列数据\n", num_rows, num_cols);

    // 步骤2：并行数据预处理 - 填充缺失值
    fillna_zero(data, num_rows, num_cols, 5); // 假设第5列包含缺失值（-1表示缺失）
    create_features(data, num_rows); // 特征工程（TotalSF）

    // 步骤3：串行数据预处理对比
    fillna_zero_serial(data, num_rows, num_cols, 5); // 串行版本
    create_features_serial(data, num_rows); // 串行版本

    // 步骤4：将处理后的数据保存到输出文件
    save_to_csv(OUTPUT_FILE, data, num_rows, num_cols);
    printf("处理后的数据已保存到文件: %s\n", OUTPUT_FILE);

    return 0;
}
