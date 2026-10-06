# 用已配置API翻译指定脚注，生成待验收PDF；不自行撰写译文，不自动替换正式文件。
import ast  # AST标准库，只读取翻译工具默认配置字典，不执行工具模块、不显示密钥。
import json  # JSON标准库，编码API请求、解析返回和保存核对记录。
import re  # 正则标准库，识别汉字、数字和代码围栏。
import urllib.request  # Python标准HTTP客户端，请求配置中的chat/completions接口。
from pathlib import Path  # 标准路径对象，根据脚本位置定位学习项目。
import fitz  # PyMuPDF，读取PDF文本位置、移除旧脚注并排入API译文。

root = Path(__file__).resolve().parents[2]  # 翻译审校→资料→多模态项目根目录。
translator = Path(r'G:\pdf2zh-v1.9.11-win64\pdf2zh\translator.py')  # 用户指定的翻译工具源码位置。
tree = ast.parse(translator.read_text(encoding='utf-8-sig'))  # 解析源码，utf-8-sig兼容UTF-8文件的BOM。
translator_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'OpenAIlikedTranslator')  # 找实际API配置类。
assignment = next(node for node in translator_class.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'envs' for target in node.targets))  # 找默认环境配置赋值。
settings = ast.literal_eval(assignment.value)  # literal_eval只解析字面量字典，不执行任意Python表达式。
targets = [('ViLT', 'ViLT', 1, 'While some works'), ('ViLT', 'ViLT', 3, 'Bugliarello et al.'), ('ViLT', 'ViLT', 4, 'Parameters for R50'), ('ViLT', 'ViLT', 5, 'ViT-B/32 is pre-trained'), ('ViLT', 'ViLT', 6, 'VQA score is calculated'), ('ViLT', 'ViLT', 7, 'UNITER proposed three'), ('ViLT', 'ViLT', 8, 'FLOPs and time'), ('PaLI-3', 'PaLI系列/PaLI-3', 3, 'embedding dimension was changed'), ('PaLI-3', 'PaLI系列/PaLI-3', 6, 'We removed all validation')]  # 页码为1起始，片段只用于精准识别已确认脚注。
items = []  # 保存脚注原文、坐标、项目路径，之后批量发送API。
for index, (paper, folder, page_number, marker) in enumerate(targets):  # enumerate同时给脚注稳定ID和其来源信息。
    original_path = root / folder / '原论文.pdf'  # 优先读项目内原文。
    if not original_path.exists():  # 部分项目只在papers目录有原文。
        original_path = root / 'papers' / (paper + '.pdf')  # 回退已下载原论文，不联网找新版本。
    with fitz.open(original_path) as document:  # with确保处理后关闭PDF文件句柄。
        matches = [block for block in document[page_number - 1].get_text('blocks') if marker in block[4] and block[1] > 640]  # 同时限制页底坐标，避免匹配正文相同词组。
        assert len(matches) == 1, (paper, page_number, 'footnote not unique')  # 脚注不唯一就停止，避免误删正文。
        block = matches[0]  # 每个目标只处理一个文本块；同块可能包含两条脚注。
        items.append({'id': str(index), 'paper': paper, 'folder': folder, 'page': page_number, 'bbox': list(block[:4]), 'source': block[4].strip()})  # 不记录任何API凭证。
cache_file = Path(__file__).with_name('脚注API译文.json')  # 保存API译文，排版重试可复用，避免重复收费。
if cache_file.exists():  # 若同一批已翻译，先校验缓存原文一致。
    cache = json.loads(cache_file.read_text(encoding='utf-8'))  # 读取既有译文及来源。
    assert [item['source'] for item in cache['items']] == [item['source'] for item in items]  # 原文有变则禁止用旧缓存冒充当前翻译。
    translations = cache['translations']  # 使用实际API返回过的对应译文。
else:  # 首次运行时调用用户配置服务。
    prompt = '请翻译以下论文脚注为简体中文，只返回JSON对象，格式为 {"translations":[{"id":"原ID","text":"译文"}]}。保留所有脚注编号、阿拉伯数字、模型标识、英文作者姓名、引用年份和网址；不要添加解释或遗漏内容，不要用英文自然语言替代译文。脚注编号放在各句开头。输入：' + json.dumps([{'id': item['id'], 'text': item['source']} for item in items], ensure_ascii=False)  # 请求服务翻译，不由本脚本编写中文内容。
    payload = {'model': settings['OPENAILIKED_MODEL'], 'messages': [{'role': 'user', 'content': prompt}]}  # 只发送原脚注和翻译要求，模型取已配置值。
    request = urllib.request.Request(settings['OPENAILIKED_BASE_URL'].rstrip('/') + '/chat/completions', data=json.dumps(payload).encode('utf-8'), headers={'Authorization': 'Bearer ' + settings['OPENAILIKED_API_KEY'], 'Content-Type': 'application/json'})  # 使用配置凭证，禁止打印request/settings。
    with urllib.request.urlopen(request, timeout=180) as response:  # 给API请求超时；外层进程会异步返回句柄，不阻塞聊天更新。
        result = json.load(response)  # 解析服务返回JSON，非自行生成的占位译文。
    content = result['choices'][0]['message']['content'].strip()  # 取服务实际译文文本。
    content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content)  # 兼容模型偶尔附带JSON代码围栏。
    translations = json.loads(content)['translations']  # 解析逐条译文。
    cache_file.write_text(json.dumps({'model': settings['OPENAILIKED_MODEL'], 'items': items, 'translations': translations, 'source': '用户已配置翻译API'}, ensure_ascii=False, indent=2), encoding='utf-8')  # 保存可审计译文，不保存密钥。
translated = {entry['id']: entry['text'] for entry in translations}  # id→文字字典，防止服务返回条目顺序变化。
assert set(translated) == {item['id'] for item in items}  # 不允许缺条或多条。
for item in items:  # 校验译文含中文且主要数字未改变，语义仍需人工复核。
    text = translated[item['id']]  # 当前对应译文。
    assert re.search('[\u4e00-\u9fff]', text), item['id']  # 确认不是整段仍英文。
    assert all(number in text for number in re.findall(r'\d+', item['source'])), (item['id'], 'numeric mismatch')  # 所有源文数字片段应保留，含编号/年份/规模。
records = []  # 排版结果记录。
for folder in dict.fromkeys(item['folder'] for item in items):  # 按项目去重，dict保留首次出现顺序。
    selected = [item for item in items if item['folder'] == folder]  # 当前项目脚注集合。
    output_directory = root / folder / '资料/脚注修复待验收'  # 独立输出，不自动覆盖正式PDF。
    output_directory.mkdir(parents=True, exist_ok=True)  # 创建待验收目录。
    for filename in ['中文论文.pdf', '双语论文.pdf']:  # 同时生成mono与dual补丁，双语原文页保持。
        with fitz.open(root / folder / filename) as document:  # 打开当前正式文件。
            for item in selected:  # 同页多项时逐项处理，坐标来自原文脚注。
                page_index = item['page'] - 1  # mono页码转为0起始索引。
                if filename == '双语论文.pdf':  # 双语每对页包含原文与译文。
                    pair = [2 * page_index, 2 * page_index + 1]  # 不武断假定中文在奇页或偶页。
                    page_index = max(pair, key=lambda number: len(re.findall('[\u4e00-\u9fff]', document[number].get_text())))  # 取该对中汉字多的译文页。
                    assert re.search('[\u4e00-\u9fff]', document[page_index].get_text())  # 确认译文页可辨认。
                page = document[page_index]  # 当前要排版的译文页。
                x0, y0, x1, y1 = item['bbox']  # 原脚注文字边界。
                rectangle = fitz.Rect(x0 - 0.5, y0 - 2, min(x0 + (234 if folder == 'ViLT' else 396), page.rect.width - 40), max(y1 + 1, 718 if folder == 'ViLT' else 734))  # 使用原列宽/页底空间，不碰页码。
                page.add_redact_annot(rectangle, fill=(1, 1, 1))  # 标记要移除旧文本的矩形，白色底。
                page.apply_redactions(images=0)  # 执行文字移除，不移除图片；目标区域已确认仅脚注文字。
                text = translated[item['id']]  # 获取API实际译文。
                font_size = 8.0  # 起始中文脚注字号，后续根据可用空间调小。
                spare = page.insert_textbox(rectangle, text, fontname='FootnoteChinese', fontfile=r'C:\Windows\Fonts\msyh.ttc', fontsize=font_size, lineheight=1.05)  # 微软雅黑同时支持中文和比例英数，避免内置CJK字体将Latin按全宽排列；负数表示未成功插入。
                while spare < 0 and font_size > 6.5:  # 仅缩小字号到合理下限，不剪掉内容。
                    font_size -= 0.25  # 每次减少四分之一点。
                    spare = page.insert_textbox(rectangle, text, fontname='FootnoteChinese', fontfile=r'C:\Windows\Fonts\msyh.ttc', fontsize=font_size, lineheight=1.05)  # 失败调用不写入文字，故可重试；每页复用相同字体名。
                assert spare >= 0, (folder, item['page'], 'text does not fit')  # 排不下就停止，不能交付缺字脚注。
                records.append({'folder': folder, 'file': filename, 'original_page': item['page'], 'patched_page_index': page_index, 'footnote_id': item['id'], 'font_size': font_size, 'remaining_height': spare, 'text': text})  # 记录排版条件，不含密钥。
            document.subset_fonts()  # 只嵌入本文件实际用到的字形，避免完整中文字体让文件膨胀。
            document.save(output_directory / filename, garbage=4, deflate=True)  # 写新PDF，垃圾回收/压缩不会更改页面顺序。
Path(__file__).with_name('脚注修复排版记录.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')  # 保存待验收结果。
print('API-translated footnotes staged:', len(items), '; PDFs:', len({item['folder'] for item in items}) * 2)  # 只打印数量，后续需要视觉/语义复核。
