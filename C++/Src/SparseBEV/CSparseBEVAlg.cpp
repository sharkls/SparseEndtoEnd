/*******************************************************
 文件名：CSparseBEVAlg.cpp
 作者：sharkls
 描述：SparseBEV算法主类实现
 版本：v1.0
 日期：2025-06-18
 *******************************************************/

#include "CSparseBEVAlg.h"

// 加载指定路径的conf配置文件并将其反序列化（解析prototext文件）
bool SparseBEVConfig::loadFromFile(const std::string& path) 
{
    std::ifstream input(path);
    if (!input) {
        LOG(ERROR) << "Failed to open config file: " << path;
        return false;
    }
    std::stringstream buffer;
    buffer << input.rdbuf();
    std::string content = buffer.str();
    if (!google::protobuf::TextFormat::ParseFromString(content, &m_config)) {
        LOG(ERROR) << "Failed to parse protobuf config file: " << path;
        return false;
    }
    return true;
}

CSparseBEVAlg::CSparseBEVAlg(const std::string& exePath)
    : m_exePath(exePath)
    , m_pConfig(std::make_shared<SparseBEVConfig>())
    , m_currentInput(nullptr)
    , m_callbackHandle(nullptr)
{
}

CSparseBEVAlg::~CSparseBEVAlg()
{
    m_moduleChain.clear();
}

bool CSparseBEVAlg::initAlgorithm(CSelfAlgParam* p_pAlgParam, const AlgCallback& alg_cb, void* hd)
{   
    // 1. 检查参数
    if (!p_pAlgParam) {
        LOG(ERROR) << "Algorithm parameters is null";
        return false;
    }

    // 2. 保存回调函数和句柄
    m_algCallback = alg_cb;
    m_callbackHandle = hd;

    // 3. 构建配置文件路径
    std::filesystem::path exePath(m_exePath);
    LOG(INFO) << "m_exePath : " << exePath.parent_path().string();
    std::string configPath = (exePath.parent_path() / "Configs"/ "Alg" / "SparseBEVConfig.conf").string();

    // 4. 加载配置文件
    if (!loadConfig(configPath)) {
        LOG(ERROR) << "Failed to load config file: " << configPath;
        return false;
    }

    m_run_status = m_pConfig->getTaskConfig().run_status();

    // 5. 初始化模块
    if (!initModules()) {
        LOG(ERROR) << "Failed to initialize modules";
        return false;
    }
    LOG(INFO) << "CSparseBEVAlg::initAlgorithm status: successs ";
    return true;
}

void CSparseBEVAlg::runAlgorithm(void* p_pSrcData)
{
    LOG(INFO) << "CSparseBEVAlg::runAlgorithm status: start ";
    // 0. 每次运行前重置结构体内容
    m_currentOutput = CAlgResult(); // 或者手动清空成员
    m_currentOutput.mapTimeStamp()[TIMESTAMP_SPARSEBEV_BEGIN] = GetTimeStamp();

    // 1. 核验输入数据是否为空
    if (!p_pSrcData) {
        LOG(ERROR) << "Input data is null";
        if (m_algCallback) {
            m_algCallback(m_currentOutput, m_callbackHandle);
        }
        return;
    }

    // 2. 输入数据赋值
    m_currentInput = static_cast<CMultiModalSrcData *>(p_pSrcData);   
    
    // 3. 执行模块链
    if (!executeModuleChain()) {
        LOG(ERROR) << "Failed to execute module chain";
        if (m_algCallback) {
            m_algCallback(m_currentOutput, m_callbackHandle);
        }
        return;
    }

    // 4. 通过回调函数返回结果
    if (m_algCallback) {
        m_algCallback(m_currentOutput, m_callbackHandle);
    }
    LOG(INFO) << "CSparseBEVAlg::runAlgorithm status: success ";

    return;
}

// 加载配置文件
bool CSparseBEVAlg::loadConfig(const std::string& configPath)
{
    return m_pConfig->loadFromFile(configPath);
}

// 初始化子模块
bool CSparseBEVAlg::initModules()
{   
    LOG(INFO) << "CSparseBEVAlg::initModules status: start ";
    auto& taskConfig = m_pConfig->getTaskConfig();
    const common::ModulesConfig& modules = taskConfig.modules_config();
    m_moduleChain.clear();

    // 遍历所有模块，按顺序实例化
    for (const common::ModuleConfig& mod : modules.modules()) {
        auto module = ModuleFactory::getInstance().createModule("SparseBEV", mod.name(), m_exePath);
        if (!module) {
            LOG(ERROR) << "Failed to create module: " << mod.name();
            return false;
        }
        // 传递可写指针
        if (!module->init((void*)taskConfig.mutable_sparsebev_config())) {
            LOG(ERROR) << "Failed to initialize module: " << mod.name();
            return false;
        }
        m_moduleChain.push_back(module);
    }
    LOG(INFO) << "CSparseBEVAlg::initModules status: success ";
    return true;
}

// 执行模块链
bool CSparseBEVAlg::executeModuleChain()
{
    void* currentData = static_cast<void *>(m_currentInput);

    for (auto& module : m_moduleChain) {
        // 设置输入数据
        module->setInput(currentData);

        // 执行模块
        module->execute();
        currentData = module->getOutput();
        if (!currentData) {
            LOG(ERROR) << "Module execution failed: " << module->getModuleName();
            return false;
        }
    }

    // 假设最后一个模块输出CAlgResult结构体
    CAlgResult* resultPtr = static_cast<CAlgResult *>(currentData);
    
    m_currentOutput = *resultPtr;
    m_currentOutput.mapTimeStamp()[TIMESTAMP_SPARSEBEV_END] = GetTimeStamp();

    // 结果穿透
    m_currentOutput.lTimeStamp() = m_currentInput->vecVideoSrcData()[0].lTimeStamp();
    if(m_currentOutput.vecFrameResult().size() > 0) 
    {   
        // 输入数据常规信息穿透
        m_currentOutput.vecFrameResult()[0].unFrameId() = m_currentInput->vecVideoSrcData()[0].unFrameId();
        m_currentOutput.vecFrameResult()[0].mapTimeStamp() = m_currentInput->vecVideoSrcData()[0].mapTimeStamp();
        m_currentOutput.vecFrameResult()[0].mapDelay() = m_currentInput->vecVideoSrcData()[0].mapDelay();
        m_currentOutput.vecFrameResult()[0].mapFps() = m_currentInput->vecVideoSrcData()[0].mapFps();

        LOG(INFO) << "原有信息穿透完毕： FrameId : " << m_currentOutput.vecFrameResult()[0].unFrameId() << ", lTimeStamp : " << m_currentOutput.lTimeStamp();

        // 独有数据填充
        m_currentOutput.vecFrameResult()[0].eDataType(DATA_TYPE_POSEALG_RESULT);                                 // 数据类型赋值
        m_currentOutput.vecFrameResult()[0].mapTimeStamp()[TIMESTAMP_POSEALG_END] = endTimeStamp;                // 姿态估计算法结束时间戳
        m_currentOutput.vecFrameResult()[0].mapDelay()[DELAY_TYPE_POSEALG] = endTimeStamp - m_currentOutput.mapTimeStamp()[TIMESTAMP_POSEALG_BEGIN];    // 姿态估计算法耗时计算
        m_currentOutput.vecFrameResult()[0].tCameraSupplement() = m_currentInput->tDisparityResult();
    }

    if(m_run_status)
    {
        visualizationResult();
    }

    return true;
} 


void CSparseBEVAlg::visualizationResult()
{
     // 1. 获取原始图像
    const auto& videoSrc = m_currentInput->vecVideoSrcData()[0];
    int width = videoSrc.usBmpWidth();
    int height = videoSrc.usBmpLength();
    int totalBytes = videoSrc.unBmpBytes();
    int channels = 0;
    if (width > 0 && height > 0) {
        channels = totalBytes / (width * height);
    }
    cv::Mat srcImage;
    if (channels == 3) {
        srcImage = cv::Mat(height, width, CV_8UC3, (void*)videoSrc.vecImageBuf().data()).clone();
    } else if (channels == 1) {
        srcImage = cv::Mat(height, width, CV_8UC1, (void*)videoSrc.vecImageBuf().data()).clone();
    } else {
        LOG(ERROR) << "Unsupported image channel: " << channels;
        return;
    }
    auto& frameResult = m_currentOutput.vecFrameResult()[0];

    // 2. 绘制检测结果
    for(const auto& obj : frameResult.vecObjectResult())
    {
        cv::Rect rect(
            cv::Point(static_cast<int>(obj.fTopLeftX()), static_cast<int>(obj.fTopLeftY())),
            cv::Point(static_cast<int>(obj.fBottomRightX()), static_cast<int>(obj.fBottomRightY()))
        );
        cv::rectangle(srcImage, rect, cv::Scalar(0, 255, 0), 2);

        // 可选：绘制类别和置信度
        std::string label = obj.strClass() + " " + std::to_string(obj.fVideoConfidence());
        cv::putText(srcImage, label, rect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);

        // 新增：绘制人体关键点
        const auto& keypoints = obj.vecKeypoints();
        // 画点
        for(const auto& kp : keypoints) {
            cv::circle(srcImage, cv::Point(static_cast<int>(kp.x()), static_cast<int>(kp.y())), 3, cv::Scalar(0,0,255), -1);
        }
    }

    // 3. 构造保存目录
    std::string visDir = (std::filesystem::path(m_exePath) / "Vis_SparseBEV_Result").string();
    if (!std::filesystem::exists(visDir)) {
        std::filesystem::create_directories(visDir);
    }

    // 4. 构造保存路径
    uint32_t frameId = frameResult.unFrameId();
    std::string savePath = visDir + "/" + std::to_string(frameId) + ".jpg";

    // 5. 保存图片
    cv::imwrite(savePath, srcImage);

    LOG(INFO) << "检测结果已保存到: " << savePath;
}