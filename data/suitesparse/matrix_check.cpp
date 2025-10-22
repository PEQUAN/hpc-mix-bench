#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <cmath>

struct SparseMatrix {
    int rows, cols;
    std::vector<std::tuple<int,int,double>> data; // store (row, col, value)

    void add(int r, int c, double v) {
        data.emplace_back(r, c, v);
    }

    int nonZeros() const {
        return data.size();
    }
};

bool loadMatrixMarket(const std::string &filename, SparseMatrix &mat) {
    std::ifstream fin(filename);
    if (!fin.is_open()) return false;

    std::string line;
    // Skip comments
    while (std::getline(fin, line)) {
        if (line[0] != '%') break;
    }

    // Read size
    int rows, cols, nnz;
    std::sscanf(line.c_str(), "%d %d %d", &rows, &cols, &nnz);
    mat.rows = rows;
    mat.cols = cols;

    // Read entries
    int r, c;
    double v;
    while (fin >> r >> c >> v) {
        mat.add(r-1, c-1, v); // convert to 0-based index
    }

    return true;
}

// Check symmetry
bool isSymmetric(const SparseMatrix &mat) {
    if(mat.rows != mat.cols) return false;

    for (size_t i = 0; i < mat.data.size(); i++) {
        int r1,c1;
        double v1;
        std::tie(r1,c1,v1) = mat.data[i];

        bool found = false;
        for (size_t j = 0; j < mat.data.size(); j++) {
            int r2,c2;
            double v2;
            std::tie(r2,c2,v2) = mat.data[j];
            if(r1==c2 && c1==r2 && std::abs(v1-v2) < 1e-12) {
                found = true;
                break;
            }
        }
        if(!found) return false;
    }
    return true;
}

// Check singularity (simple check: zero diagonal)
bool isSingular(const SparseMatrix &mat) {
    std::vector<double> diag(mat.rows, 0.0);
    for(auto &[r,c,v] : mat.data) {
        if(r==c) diag[r] = v;
    }
    for(auto &d : diag) if(std::abs(d) < 1e-12) return true;
    return false;
}

int main(int argc, char** argv) {
    if(argc < 2) {
        std::cout << "Usage: " << argv[0] << " matrix_file.mtx\n";
        return 1;
    }

    SparseMatrix mat;
    if(!loadMatrixMarket(argv[1], mat)) {
        std::cerr << "Failed to load matrix.\n";
        return 1;
    }

    std::cout << "Matrix loaded: " << mat.rows << " x " << mat.cols << "\n";

    double sparsity = 1.0 - double(mat.nonZeros()) / (mat.rows * mat.cols);
    std::cout << "Sparsity: " << sparsity*100 << "%\n";

    std::cout << "Symmetric: " << (isSymmetric(mat) ? "Yes" : "No") << "\n";

    std::cout << "Singular (zero diagonal check): " 
              << (isSingular(mat) ? "Yes" : "No") << "\n";

    return 0;
}
