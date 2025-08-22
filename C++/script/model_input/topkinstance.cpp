#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct IndexedValue {
    float value;
    std::uint32_t index;
};

// 与 C++/Src/SparseBEV8.6/Inference/Utils.cpp 中一致：仅按 value 降序
static inline bool compareIndexedValue(const IndexedValue &a, const IndexedValue &b) {
    return a.value > b.value;
}

static bool readBinFloat32(const std::string &path, std::vector<float> &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "[ERROR] Failed to open file: " << path << std::endl;
        return false;
    }
    f.seekg(0, std::ios::end);
    const std::streampos sz = f.tellg();
    f.seekg(0, std::ios::beg);
    if (sz <= 0) {
        std::cerr << "[ERROR] File is empty: " << path << std::endl;
        return false;
    }
    out.resize(static_cast<size_t>(sz) / sizeof(float));
    f.read(reinterpret_cast<char *>(out.data()), sz);
    if (!f) {
        std::cerr << "[ERROR] Failed to read file: " << path << std::endl;
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    // 默认参数
    std::string binPath = "/share/Code/SparseEnd2End/C++/Output/val_bin/sample_0_pred_class_score_1*900*10_float32.bin";
    std::uint32_t topK = 20;

    if (argc >= 2) binPath = argv[1];
    if (argc >= 3) topK = static_cast<std::uint32_t>(std::stoul(argv[2]));

    constexpr std::uint32_t B = 1;
    constexpr std::uint32_t N = 900;  // num_querys
    constexpr std::uint32_t C = 10;   // 类别数
    const size_t expectedCount = static_cast<size_t>(B) * N * C;

    std::vector<float> raw;
    if (!readBinFloat32(binPath, raw)) {
        return 1;
    }
    if (raw.size() != expectedCount) {
        std::cerr << "[ERROR] Element count mismatch. Expect " << expectedCount
                  << ", got " << raw.size() << std::endl;
        return 2;
    }

    // 计算 confidence = max(dim=-1) → [N]
    std::vector<float> confidence(N, -std::numeric_limits<float>::infinity());
    for (std::uint32_t i = 0; i < N; ++i) {
        float m = raw[i * C + 0];
        for (std::uint32_t j = 1; j < C; ++j) {
            float v = raw[i * C + j];
            if (v > m) m = v;
        }
        confidence[i] = m;
    }

    // 构造 (value, index) 并排序（仅按 value 降序，与工程实现一致）
    std::vector<IndexedValue> indexedValues(N);
    for (std::uint32_t i = 0; i < N; ++i) {
        indexedValues[i].value = confidence[i];
        indexedValues[i].index = i;
    }
    std::sort(indexedValues.begin(), indexedValues.end(), compareIndexedValue);

    if (topK > N) topK = N;

    std::cout << "[INFO] Sorted by value (desc) using compareIndexedValue, Top-" << topK << "\n";
    std::cout << std::fixed << std::setprecision(6);

    // 输出TopK索引和值
    std::cout << "Indices: ";
    for (std::uint32_t i = 0; i < topK; ++i) {
        std::cout << indexedValues[i].index << (i + 1 == topK ? '\n' : ' ');
    }

    std::cout << "Values : ";
    for (std::uint32_t i = 0; i < topK; ++i) {
        std::cout << indexedValues[i].value << (i + 1 == topK ? '\n' : ' ');
    }

    // 额外：打印TopK中相等值的分组（可选）
    std::uint32_t equalCount = 0;
    for (std::uint32_t i = 1; i < topK; ++i) {
        if (indexedValues[i].value == indexedValues[i - 1].value) {
            ++equalCount;
        }
    }
    if (equalCount > 0) {
        std::cout << "Equal-value groups within Top-" << topK << ":\n";
        float curVal = indexedValues[0].value;
        std::vector<std::uint32_t> group{indexedValues[0].index};
        for (std::uint32_t i = 1; i < topK; ++i) {
            if (indexedValues[i].value == curVal) {
                group.push_back(indexedValues[i].index);
            } else {
                if (group.size() > 1) {
                    std::cout << "  value=" << curVal << " -> indices=[";
                    for (size_t t = 0; t < group.size(); ++t) {
                        std::cout << group[t] << (t + 1 == group.size() ? ']' : ' ');
                    }
                    std::cout << '\n';
                }
                curVal = indexedValues[i].value;
                group.clear();
                group.push_back(indexedValues[i].index);
            }
        }
        if (group.size() > 1) {
            std::cout << "  value=" << curVal << " -> indices=[";
            for (size_t t = 0; t < group.size(); ++t) {
                std::cout << group[t] << (t + 1 == group.size() ? ']' : ' ');
            }
            std::cout << '\n';
        }
    }

    return 0;
}
