import sys
import logging
import logging.config
# ================== Logger ================================
def Logger(file_name='/root/trainlog/log'):
      formatter = logging.Formatter(fmt='%(asctime)s %(module)s,line: %(lineno)d %(levelname)8s | %(message)s',
                                    datefmt='%Y/%m/%d %H:%M:%S')
      logging.basicConfig(filename = '%s.log' %(file_name),format= '%(asctime)s %(module)s,line: %(lineno)d %(levelname)8s | %(message)s',
                                    datefmt='%Y/%m/%d %H:%M:%S', filemode = 'a', level = logging.INFO)
      log_object = logging.getLogger()
      log_object.setLevel(logging.DEBUG)
      # console printer
      screen_handler = logging.StreamHandler(stream=sys.stdout) 
      screen_handler.setFormatter(formatter)
      logging.getLogger().addHandler(screen_handler)
      return log_object
    # =======================================================
