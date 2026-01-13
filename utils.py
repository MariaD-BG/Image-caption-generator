import re
def strip_syntax(txt:str) ->str:
    cleaned_txt = re.sub(r'[^a-zA-Z ]', '', txt)
    return cleaned_txt