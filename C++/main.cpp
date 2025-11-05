#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <google/protobuf/text_format.h>
#include "TestConfig_conf.pb.h"
#include "log.h"

// 前向声明各个测试单元的主函数
// void main_sparse_bev();
// void main_sparse_bev_v2();
void main_sparse_bev_8_6();
void main_sparse_4d();

// 测试单元基类
class TestUnit {
public:
    virtual ~TestUnit() = default;
    virtual void run() = 0;
};


// // SparseBEV测试单元
// class SparseBEVTest : public TestUnit {
// public:
//     void run() override {
//         try {
//             main_sparse_bev();
//         } catch (const std::exception& e) {
//             LOG(ERROR) << "SparseBEV测试错误: " << e.what();
//         }
//     }
// };

// // SparseBEV v2测试单元（用于asset数据）
// class SparseBEVTestV2 : public TestUnit {
// public:
//     void run() override {
//         try {
//             main_sparse_bev_v2();
//         } catch (const std::exception& e) {
//             LOG(ERROR) << "SparseBEV v2测试错误: " << e.what();
//         }
//     }
// };

// SparseBEV 8.6测试单元（用于TensorRT 8.6）
class SparseBEVTest8_6 : public TestUnit {
public:
    void run() override {
        try {
            main_sparse_bev_8_6();
        } catch (const std::exception& e) {
            LOG(ERROR) << "SparseBEV 8.6测试错误: " << e.what();
        }
    }
};

// Sparse4D测试单元
class Sparse4DTest : public TestUnit {
public:
    void run() override {
        try {
            main_sparse_4d();
        } catch (const std::exception& e) {
            LOG(ERROR) << "Sparse4D测试错误: " << e.what();
        }
    }
};

// 创建测试单元工厂
std::unique_ptr<TestUnit> createTestUnit(const std::string& task) {
    // if (task == "SparseBEV") {
    //     return std::make_unique<SparseBEVTest>();
    // } else if (task == "SparseBEV_v2") {
    //     return std::make_unique<SparseBEVTestV2>();
    // } else 
    if (task == "SparseBEV_8_6") {
        return std::make_unique<SparseBEVTest8_6>();
    } else if (task == "Sparse4D") {
        return std::make_unique<Sparse4DTest>();
    }
    return nullptr;
}

int main() {
    try {
        // 读取配置文件
        std::string config_path = "/share/Code/Sparse4dE2E/C++/Output/Configs/Alg/TestConfig.conf";
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            LOG(ERROR) << "无法打开配置文件: " << config_path;
            return -1;
        }

        std::string config_content((std::istreambuf_iterator<char>(config_file)),
                                 std::istreambuf_iterator<char>());
        config_file.close();

        // 解析配置文件
        test::TestConfig test_config;
        if (!google::protobuf::TextFormat::ParseFromString(config_content, &test_config)) {
            LOG(ERROR) << "解析配置文件失败";
            return -1;
        }

        // 获取任务名称
        std::string task = test_config.task();
        if (task.empty()) {
            LOG(ERROR) << "配置文件中未指定任务";
            return -1;
        }

        // 执行测试单元
        LOG(INFO) << "开始执行测试单元: " << task;
        auto test_unit = createTestUnit(task);
        if (test_unit) {
            test_unit->run();
        } else {
            LOG(ERROR) << "未知的测试单元: " << task;
            return -1;
        }

    } catch (const std::exception& e) {
        LOG(ERROR) << "错误: " << e.what();
        return -1;
    }

    return 0;
} 