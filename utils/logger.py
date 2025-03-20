import os
import datetime
from colorama import Fore, Style, init
from tqdm import tqdm

init(autoreset=True)

class Logger:
  COLORS = {
    'INFO': Fore.CYAN,
    'SUCCESS': Fore.GREEN,
    'WARNING': Fore.YELLOW,
    'ERROR': Fore.RED,
    'DEBUG': Fore.MAGENTA,
  }
  
  def __init__(self):
    pass
  
  def _get_timestamp(self):
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  
  def _log(self, level, message, *args, timestamp=False, use_tqdm=True, **kwargs):
    if args or kwargs:
      message = message.format(*args, **kwargs)
    
    color = self.COLORS.get(level, Fore.WHITE)
    
    if timestamp:
      time_str = self._get_timestamp()
      prefix = f"{color}[{time_str}] [{level}]"
    else:
      prefix = f"{color}[{level}]"
    
    formatted_message = f"{prefix} {message}{Style.RESET_ALL}"
    
    if use_tqdm:
      tqdm.write(formatted_message)
    else:
      print(formatted_message)
  
  def text(self, message, *args, timestamp=False, use_tqdm=True, **kwargs):
    if args or kwargs: 
      message = message.format(*args, **kwargs)
    
    if timestamp:
      time_str = self._get_timestamp()
      formatted_message = f"[{time_str}] {message}"
    else:
      formatted_message = f"{message}"
    
    if use_tqdm:
      tqdm.write(formatted_message)
    else:
      print(formatted_message)
  
  def info(self, message, *args, timestamp=False, use_tqdm=True, **kwargs):
    self._log("INFO", message, *args, timestamp=timestamp, use_tqdm=use_tqdm, **kwargs)
  
  def success(self, message, *args, timestamp=False, use_tqdm=True, **kwargs):
    self._log("SUCCESS", message, *args, timestamp=timestamp, use_tqdm=use_tqdm, **kwargs)
  
  def warning(self, message, *args, timestamp=False, use_tqdm=True, **kwargs):
    self._log("WARNING", message, *args, timestamp=timestamp, use_tqdm=use_tqdm, **kwargs)
  
  def error(self, message, *args, timestamp=False, use_tqdm=True, **kwargs):
    self._log("ERROR", message, *args, timestamp=timestamp, use_tqdm=use_tqdm, **kwargs)
  
  def debug(self, message, *args, timestamp=False, use_tqdm=True, **kwargs):
    self._log("DEBUG", message, *args, timestamp=timestamp, use_tqdm=use_tqdm, **kwargs)

  def draw_header(
    self, 
    heading,
    version = "1",
    width = 50,
    clear_screen = True
  ):
    if clear_screen:
      self.clear_screen()

    border_color = Fore.WHITE
    heading_color = Fore.WHITE
    content_color = Fore.WHITE
    project_color = Fore.WHITE
    
    borders = {
      'tl': '┌', 'tr': '┐', 'bl': '└', 'br': '┘',
      'h': '─', 'v': '│', 'lj': '├', 'rj': '┤'
    }
    
    def format_border(left, middle, right):
      return f"{border_color}{left}{middle * (width-2)}{right}{Style.RESET_ALL}"
    
    def format_line(text, color, centered=True, padding_char=' '):
      max_len = width - 4
      if len(text) > max_len:
        return wrap_text(text, color, centered)
      
      if centered:
        padding = max(0, width - len(text) - 4)
        left_pad = padding // 2
        right_pad = padding - left_pad
        padding_str = f"{padding_char * left_pad}{text}{padding_char * right_pad}"
      else:
        padding_str = f"{text}{padding_char * (max_len - len(text))}"
      
      return [f"{border_color}{borders['v']}{Style.RESET_ALL} {color}{padding_str}{Style.RESET_ALL} {border_color}{borders['v']}{Style.RESET_ALL}"]
    
    def wrap_text(text, color, centered=True):
      lines = []
      max_text_length = width - 4
      
      if len(text) <= max_text_length:
        return format_line(text, color, centered)
      
      words = text.split()
      current_line = ""
      for word in words:
        if len(current_line) + len(word) + 1 <= max_text_length:
          current_line = word if not current_line else f"{current_line} {word}"
        else:
          lines.extend(format_line(current_line, color, centered))
          current_line = word
      
      if current_line:
        lines.extend(format_line(current_line, color, centered))
      
      return lines
    
    top_border = format_border(borders['tl'], borders['h'], borders['tr'])
    bottom_border = format_border(borders['bl'], borders['h'], borders['br'])
    separator = format_border(borders['lj'], borders['h'], borders['rj'])
    
    output_lines = [top_border]
    
    # header title
    heading_with_version = f"{heading} (v{version})"
    output_lines.extend(wrap_text(heading_with_version, heading_color))
    output_lines.append(separator)

    # content lines
    content_items = [
      f"Dilawaiz Sarwat",
      f"» 23015919-007@uog.edu.pk",
      f"» MPhil Computer Science",
      f"» Department of Computer Science",
      f"» University of Gujrat, Hafiz Hayat Campus"
    ]
    
    for item in content_items:
      output_lines.extend(format_line(item, content_color, centered=False))
    
    output_lines.append(separator)
    
    # project title
    project_title = "A Deep Learning Framework for Prostate Cancer Analysis with Weak Supervision"
    output_lines.extend(wrap_text(project_title, project_color))
    
    output_lines.append(bottom_border)
    
    for line in output_lines: print(line)
    self.empty_line()

  def empty_line(self, use_tqdm=True):
    if use_tqdm:
      tqdm.write("")
    else:
      print()

  def clear_screen(self):
    os.system('cls' if os.name=='nt' else 'clear')