import re
import underthesea
replace_list = {
    'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ', 'òe': 'oè', 'óe': 'oé','ỏe': 'oẻ',
    'õe': 'oẽ', 'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ','ụy': 'uỵ', 'uả': 'ủa',
    'ả': 'ả', 'ố': 'ố', 'u´': 'ố','ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
    'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề','ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
    'ẻ': 'ẻ', 'àk': u' à ','aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ','ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á',
    #Quy các icon về 2 loại emoj: Tích cực hoặc tiêu cực
    "👹": "tiêu cực", "👻": "tích cực", "💃": "tích cực",'🤙': ' tích cực ', '👍': ' tích cực ',
    "💄": "tích cực", "💎": "tích cực", "💩": "tích cực","😕": "tiêu cực", "😱": "tiêu cực", "😸": "tích cực",
    "😾": "tiêu cực", "🚫": "tiêu cực",  "🤬": "tiêu cực","🧚": "tích cực", "🧡": "tích cực",'🐶':' tích cực ',
    '👎': ' tiêu cực ', '😣': ' tiêu cực ','✨': ' tích cực ', '❣': ' tích cực ','☀': ' tích cực ',
    '♥': ' tích cực ', '🤩': ' tích cực ', 'like': ' tích cực ', '💌': ' tích cực ',
    '🤣': ' tích cực ', '🖤': ' tích cực ', '🤤': ' tích cực ', ':(': ' tiêu cực ', '😢': ' tiêu cực ',
    '❤': ' tích cực ', '😍': ' tích cực ', '😘': ' tích cực ', '😪': ' tiêu cực ', '😊': ' tích cực ',
    '?': ' ? ', '😁': ' tích cực ', '💖': ' tích cực ', '😟': ' tiêu cực ', '😭': ' tiêu cực ',
    '💯': ' tích cực ', '💗': ' tích cực ', '♡': ' tích cực ', '💜': ' tích cực ', '🤗': ' tích cực ',
    '^^': ' tích cực ', '😨': ' tiêu cực ', '☺': ' tích cực ', '💋': ' tích cực ', '👌': ' tích cực ',
    '😖': ' tiêu cực ', '😀': ' tích cực ', ':((': ' tiêu cực ', '😡': ' tiêu cực ', '😠': ' tiêu cực ',
    '😒': ' tiêu cực ', '🙂': ' tích cực ', '😏': ' tiêu cực ', '😝': ' tích cực ', '😄': ' tích cực ',
    '😙': ' tích cực ', '😤': ' tiêu cực ', '😎': ' tích cực ', '😆': ' tích cực ', '💚': ' tích cực ',
    '✌': ' tích cực ', '💕': ' tích cực ', '😞': ' tiêu cực ', '😓': ' tiêu cực ', '️🆗️': ' tích cực ',
    '😉': ' tích cực ', '😂': ' tích cực ', ':v': '  tích cực ', '=))': '  tích cực ', '😋': ' tích cực ',
    '💓': ' tích cực ', '😐': ' tiêu cực ', ':3': ' tích cực ', '😫': ' tiêu cực ', '😥': ' tiêu cực ',
    '😃': ' tích cực ', '😬': ' 😬 ', '😌': ' 😌 ', '💛': ' tích cực ', '🤝': ' tích cực ', '🎈': ' tích cực ',
    '😗': ' tích cực ', '🤔': ' tiêu cực ', '😑': ' tiêu cực ', '🔥': ' tiêu cực ', '🙏': ' tiêu cực ',
    '🆗': ' tích cực ', '😻': ' tích cực ', '💙': ' tích cực ', '💟': ' tích cực ',
    '😚': ' tích cực ', '❌': ' tiêu cực ', '👏': ' tích cực ', ';)': ' tích cực ', '<3': ' tích cực ',
    '🌝': ' tích cực ',  '🌷': ' tích cực ', '🌸': ' tích cực ', '🌺': ' tích cực ',
    '🌼': ' tích cực ', '🍓': ' tích cực ', '🐅': ' tích cực ', '🐾': ' tích cực ', '👉': ' tích cực ',
    '💐': ' tích cực ', '💞': ' tích cực ', '💥': ' tích cực ', '💪': ' tích cực ',
    '💰': ' tích cực ',  '😇': ' tích cực ', '😛': ' tích cực ', '😜': ' tích cực ',
    '🙃': ' tích cực ', '🤑': ' tích cực ', '🤪': ' tích cực ','☹': ' tiêu cực ',  '💀': ' tiêu cực ',
    '😔': ' tiêu cực ', '😧': ' tiêu cực ', '😩': ' tiêu cực ', '😰': ' tiêu cực ', '😳': ' tiêu cực ',
    '😵': ' tiêu cực ', '😶': ' tiêu cực ', '🙁': ' tiêu cực ',
    #Chuẩn hóa 1 số sentiment words/English words
    ':))': '  tích cực ', ':)': ' tích cực ', 'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
    'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ':  ' ok ',' okay':' ok ','okê':' ok ',
    ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',
    '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u' tích cực ',
    'kg ': u' không ','not': u' không ', u' kg ': u' không ', '"k ': u' không ',' kh ':u' không ','kô':u' không ','hok':u' không ',' kp ': u' không phải ',u' kô ': u' không ', '"ko ': u' không ', u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
    'he he': ' tích cực ','hehe': ' tích cực ','hihi': ' tích cực ', 'haha': ' tích cực ', 'hjhj': ' tích cực ',
    ' lol ': ' tiêu cực ',' cc ': ' tiêu cực ','cute': u' dễ thương ','huhu': ' tiêu cực ', ' vs ': u' với ', 'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',
    ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',
    'đc': u' được ','authentic': u' chuẩn chính hãng ',u' aut ': u' chuẩn chính hãng ', u' auth ': u' chuẩn chính hãng ', 'thick': u' tích cực ', 'store': u' cửa hàng ',
    'shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tích cực ','god': u' tích cực ','wel done':' tích cực ', 'good': u' tích cực ', 'gút': u' tích cực ',
    'sấu': u' xấu ','gut': u' tích cực ', u' tot ': u' tích cực ', u' nice ': u' tích cực ', 'perfect': 'rất tích cực', 'bt': u' bình thường ',
    'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
    'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng','chat':' chất ', 'excelent': 'hoàn hảo', 'bad': 'tiêu cực','fresh': ' tươi ','sad': ' tiêu cực ',
    'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ','quickly': u' nhanh ', 'quick': u' nhanh ','fast': u' nhanh ','delivery': u' giao hàng ',u' síp ': u' giao hàng ',
    'beautiful': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',u' order ': u' đặt hàng ',
    'chất lg': u' chất lượng ',u' sd ': u' sử dụng ',u' dt ': u' điện thoại ',u' nt ': u' nhắn tin ',u' tl ': u' trả lời ',u' sài ': u' xài ',u'bjo':u' bao giờ ',
    'thik': u' thích ',u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',u'quả ng ':u' quảng  ',
    'dep': u' đẹp ',u' xau ': u' xấu ','delicious': u' ngon ', u'hàg': u' hàng ', u'qủa': u' quả ','bk':'biết',
    'iu': u' yêu ','fake': u' giả mạo ', 'trl': 'trả lời', '><': u' tích cực ', 'dthoai':'điện thoại','thegioididong':'thế giới di động', 'đt': 'điện thoại','dt': 'điện thoại',
    ' por ': u' tiêu cực ',' poor ': u' tiêu cực ', 'ib':u' nhắn tin ', 'rep':u' trả lời ',u'fback':' feedback ','fedback':' feedback ','zin':'tích cực','fb': 'mạng xã hội','lag':'tiêu cực',
    'hazzz': 'tiêu cực','test': 'thử','bit': 'biết', 'ak':'à','noiz':'nói','rats':'rất','j':'gì','diss':'tiêu cực',
    '4 sao': 'tích cực','5 sao': 'tích cực','1 sao': 'tiêu cực', 'phên': 'tích cực','kog': 'không', 'mia': 'mua', 'ja':'giá','way':'quay', 'zay':'vậy'}

# with open('C:/Users/FPTSHOP/OneDrive/Documents/SAV/CrawlData/mobileData/teencode.txt', encoding='utf-8') as f:
#     for pair in f.readlines():
#         key, value = pair.split('\t')
#         replace_list[key] = value.strip()
        
def normalize_text(text):
    # xóa các ký tự kéo dài: tốtttttttt
    text = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), text, flags=re.IGNORECASE)
    
    # chuẩn hóa chữ thường
    text = text.lower()

    # Format lại ký tự đặc biệt, chữ viết tắt, emoji
    for k, v in replace_list.items():
        text = text.replace(k, v)   
    return text

def remove_unnecessary_characters(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# word segmentation
def preprocess(text):
    text = normalize_text(text)
    tokens = text.split()
    
    text= underthesea.word_tokenize(" ".join(tokens), format="text")
    
    text = remove_unnecessary_characters(text)
    return text
