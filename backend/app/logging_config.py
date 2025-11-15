"""
Logging Configuration for Production Backend
Structured logging with different levels and handlers
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json


# ============================================================================
# LOG FORMATTERS
# ============================================================================

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        if hasattr(record, 'process_name'):
            log_data['process_name'] = record.process_name
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format message with color
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(
    log_level: str = 'INFO',
    log_dir: Optional[str] = 'logs',
    log_to_file: bool = True,
    log_to_console: bool = True,
    json_format: bool = False,
    colored_console: bool = True
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (None to disable file logging)
        log_to_file: Enable file logging
        log_to_console: Enable console logging
        json_format: Use JSON format for logs
        colored_console: Use colored output for console
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('calorflow')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        
        if json_format:
            console_handler.setFormatter(JSONFormatter())
        elif colored_console:
            console_formatter = ColoredFormatter(
                '%(levelname)s | %(asctime)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
        else:
            console_formatter = logging.Formatter(
                '%(levelname)s | %(asctime)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
        
        logger.addHandler(console_handler)
    
    # File handlers
    if log_to_file and log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # General log file
        file_handler = logging.FileHandler(
            log_path / f'calorflow_{datetime.now().strftime("%Y%m%d")}.log'
        )
        file_handler.setLevel(logging.DEBUG)
        
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_formatter = logging.Formatter(
                '%(levelname)s | %(asctime)s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
        
        # Error log file
        error_handler = logging.FileHandler(
            log_path / f'calorflow_errors_{datetime.now().strftime("%Y%m%d")}.log'
        )
        error_handler.setLevel(logging.ERROR)
        
        if json_format:
            error_handler.setFormatter(JSONFormatter())
        else:
            error_formatter = logging.Formatter(
                '%(levelname)s | %(asctime)s | %(name)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s\n'
                'Exception: %(exc_info)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            error_handler.setFormatter(error_formatter)
        
        logger.addHandler(error_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f'calorflow.{name}')


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

class LogContext:
    """Context manager for adding context to logs"""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def log_execution_time(logger: logging.Logger, operation: str):
    """
    Decorator to log execution time
    
    Args:
        logger: Logger instance
        operation: Operation name
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start = time.time()
            logger.info(f"Starting {operation}")
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                logger.info(f"Completed {operation} in {elapsed:.2f}s")
                return result
            except Exception as e:
                elapsed = time.time() - start
                logger.error(f"Failed {operation} after {elapsed:.2f}s: {str(e)}")
                raise
        
        return wrapper
    return decorator


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls with arguments
    
    Args:
        logger: Logger instance
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args[:2]}..., kwargs={list(kwargs.keys())}")
            result = func(*args, **kwargs)
            logger.debug(f"Completed {func.__name__}")
            return result
        return wrapper
    return decorator


# ============================================================================
# PRE-CONFIGURED LOGGERS
# ============================================================================

# Default logger (can be reconfigured)
default_logger = setup_logging(
    log_level='INFO',
    log_dir='logs',
    log_to_file=True,
    log_to_console=True,
    json_format=False,
    colored_console=True
)


# Production logger (JSON format, file only)
def setup_production_logging():
    """Setup production logging configuration"""
    return setup_logging(
        log_level='WARNING',
        log_dir='logs',
        log_to_file=True,
        log_to_console=False,
        json_format=True,
        colored_console=False
    )


# Development logger (colored console, debug level)
def setup_development_logging():
    """Setup development logging configuration"""
    return setup_logging(
        log_level='DEBUG',
        log_dir='logs',
        log_to_file=True,
        log_to_console=True,
        json_format=False,
        colored_console=True
    )


if __name__ == '__main__':
    # Test logging
    logger = setup_development_logging()
    
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    # Test context
    with LogContext(logger, request_id='123', process_name='FCC'):
        logger.info("Message with context")
    
    print("\nLogging configuration ready!")
