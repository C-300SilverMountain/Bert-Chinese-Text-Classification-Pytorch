import logging
import os
import sys
import time

import unicodedata


def get_log():
    logger = logging.getLogger('ner_model')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    fmt = "{asctime} {levelname:^8s} {filename}:{lineno:<4}  {message}"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S", style="{")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

logger = get_log()
class Timer:
    def __init__(self, id_msg='', if_print=True):
        self.dt = time.time()
        self.start_msg = id_msg
        self.if_print = if_print

    def print_time(self, msg=''):
        if self.if_print:
            print(f'{self.start_msg}: {msg} cost time: {(time.time() - self.dt) * 1000:.0f}ms')
        self.dt = time.time()


class OffsetMapping:
    def __init__(self):
        self._do_lower_case = True

    @staticmethod
    def stem(token):
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    @staticmethod
    def _is_control(ch):
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    def rematch(self, text, tokens):
        if self._do_lower_case:
            text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            if self._do_lower_case:
                ch = unicodedata.normalize('NFD', ch)
                ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([c for c in ch if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            if self._is_special(token):
                token_mapping.append([offset])
                offset += 1
            else:
                token = self.stem(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                token_mapping.append(char_mapping[start:end])
                offset = end

        return token_mapping

def extract_entity(text, entity_idx, text_start_id, text_mapping):
    start_split = text_mapping[entity_idx[0] - text_start_id] if entity_idx[
                                                                     0] - text_start_id < len(text_mapping) and \
                                                                 entity_idx[0] - text_start_id >= 0 else []
    end_split = text_mapping[entity_idx[1] - text_start_id] if (len(text_mapping) > entity_idx[1] - text_start_id >= 0) else []
    entity = ''
    if start_split != [] and end_split != []:
        entity = text[start_split[0]:end_split[-1] + 1]
    return entity


def peek_folder(folder_path):
    # folder_path = args.parent_path
    folder_content = os.listdir(folder_path)

    for content in folder_content:
        filename = content
        file_size = 0
        try:
            file_size = os.stat(folder_path+'/'+filename).st_size
        except Exception as e:
            pass
        print(f'filename:{content}, size:{file_size}bytes')
        # print("文件大小: {} bytes".format(file_size))