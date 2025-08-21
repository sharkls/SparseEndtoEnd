bool ErrorHandler::initializeFromConfig(const std::string& config_path) {
    ErrorHandlerConfigData config_data;
    if (!ErrorHandlerConfigParser::parseConfigFile(config_path, config_data)) {
        return false;
    }
    
    return initializeFromConfigData(config_data);
}

bool ErrorHandler::initializeFromConfigData(const ErrorHandlerConfigData& config_data) {
    std::lock_guard<std::mutex> lock(m_mutex);
    
    // 验证配置
    if (!ErrorHandlerConfigParser::validateConfig(config_data)) {
        return false;
    }
    
    // 保存配置数据
    m_config_data = config_data;
    
    // 更新配置
    m_config.enable_error_handling = config_data.global_config.enable_error_handling;
    m_config.enable_error_recovery = config_data.global_config.enable_error_recovery;
    m_config.enable_error_monitoring = config_data.global_config.enable_error_monitoring;
    m_config.max_error_history = config_data.global_config.max_error_history;
    m_config.log_level = config_data.global_config.log_level;
    
    // 初始化错误恢复器
    if (m_config.enable_error_recovery) {
        m_error_recovery = std::make_unique<ErrorRecovery>();
        m_error_recovery->setRecoveryStrategies(config_data.recovery_strategies);
    }
    
    // 初始化错误监控器
    if (m_config.enable_error_monitoring) {
        m_error_monitor = std::make_unique<ErrorMonitor>();
        m_error_monitor->setConfig(config_data.monitor_config);
        m_error_monitor->setErrorThresholds(config_data.error_thresholds);
        m_error_monitor->setModuleThresholds(config_data.module_thresholds);
        m_error_monitor->setAlertConfig(config_data.alert_config);
    }
    
    return true;
}

ErrorHandlerConfig ErrorHandler::getConfig() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_config;
}

ErrorHandlerConfigData ErrorHandler::getConfigData() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_config_data;
} 