import logging
import colorlog

log_colors_config = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "bold_red",
}


def set_logger(log_path, save_file=False):
    # 创建logger
    logger = logging.getLogger("logger_name")
    # 创建控制台处理器
    console_handler = logging.StreamHandler()

    # 创建文件处理器
    file_formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s]: %(message)s", datefmt="%Y-%m-%d::%H:%M:%S"  # 设置文件处理器格式：时间、日志级别、消息
    )

    # 创建控制台格式化器
    console_formatter = colorlog.ColoredFormatter(
        fmt="%(log_color)s[%(asctime)s] [%(levelname)s]:%(reset)s %(message)s",  # 设置控制台格式：时间、日志级别、消息
        datefmt="%Y-%m-%d::%H:%M:%S",  # 设置时间格式
        log_colors=log_colors_config,  # 设置日志颜色
    )

    # 设置控制台处理器格式化器
    console_handler.setFormatter(console_formatter)

    # 如果logger没有处理器，则添加控制台处理器
    if not logger.handlers:
        # 添加控制台处理器
        logger.addHandler(console_handler)

    # 关闭控制台处理器
    console_handler.close()

    # 创建文件处理器
    file_handler = None
    if save_file:
        # 创建文件处理器
        file_handler = logging.FileHandler(filename=log_path, mode="a", encoding="utf8")
        # 设置文件处理器格式化器
        file_handler.setFormatter(file_formatter)
        # 添加文件处理器
        if not logger.handlers:
            logger.addHandler(file_handler)
        # 关闭文件处理器
        file_handler.close()

    # 返回logger, file_handler, console_handler
    return logger, file_handler, console_handler


def logger_wrapper(
    log_path="",
    save_file=False,
    level1=logging.DEBUG,
    level2=logging.DEBUG,
    level3=logging.DEBUG,
):
    logger, file_handler, console_handler = set_logger(log_path, save_file)

    logger.setLevel(level1)
    console_handler.setLevel(level2)
    if save_file:
        file_handler.setLevel(level3)
    return logger, file_handler, console_handler
