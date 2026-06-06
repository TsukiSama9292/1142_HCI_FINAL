import re
from typing import Literal

FilterAction = Literal["noise", "number", "empty"]
RuleResult = tuple[str, str] | None


NOISE_KEYWORDS = [
    "簽到", "簽", "刷", "工商", "台北", "臺北", "台大", "臺大",
    "test", "測試", "廣告",
]

NUMBER_SLANG: dict[re.Pattern, str] = {
    re.compile(r"^233+$"): "amusement",
    re.compile(r"^666+$"): "admiration",
    re.compile(r"^555+$"): "sadness",
}

DIRECT_EMOTION_RULES: list[tuple[re.Pattern, str]] = [
    # A - Admiration 讚賞
    (re.compile(r"(太神|大神|神仙|跪了|膜拜|天花板|頂尖)"), "admiration"),
    (re.compile(r"(作畫|聲優)[^，]*[神炸爆]"), "admiration"),
    (re.compile(r"(經費爆炸|經費在燃燒)"), "admiration"),
    (re.compile(r"這[^，]*太強"), "admiration"),

    # B - Amusement 有趣
    (re.compile(r"(笑死|笑爛|笑到|嘴角上揚)"), "amusement"),
    (re.compile(r"(XD|xD|xd|233|www|草)"), "amusement"),
    (re.compile(r"哈哈"), "amusement"),
    (re.compile(r"肚子好痛"), "amusement"),
    (re.compile(r"有夠好笑"), "amusement"),

    # C - Approval 認可
    (re.compile(r"(說得好|\+1|確實|同意|合理|正解|無誤|真男人)"), "approval"),
    (re.compile(r"沒錯"), "approval"),
    (re.compile(r"^(👍|👏|🙌|💪|🔥|推)"), "approval"),

    # D - Caring 關心
    (re.compile(r"(保重|小心|沒事吧|還好嗎|辛苦了|加油|撐下去)"), "caring"),

    # E - Desire 慾望
    (re.compile(r"(好想要|我也要|羨慕|好想|想要|此處有本)"), "desire"),

    # F - Excitement 興奮
    (re.compile(r"(燃爆|熱血|高潮了|來了來了|上啊|帥啊|666)"), "excitement"),
    (re.compile(r"太[^，]*(帥|猛)"), "excitement"),

    # G - Gratitude 感激
    (re.compile(r"(謝謝|感謝|感激|好人一生平安)"), "gratitude"),

    # H - Joy 喜悅
    (re.compile(r"(太棒|太好|好棒|開心|舒服了|好爽|愉悅)"), "joy"),

    # I - Love 愛
    (re.compile(r"(婆爆|我婆|我老公|我推|老婆|我愛|暈了)"), "love"),
    (re.compile(r"好[^，]*(可愛|萌)"), "love"),

    # J - Optimism 樂觀
    (re.compile(r"(相信|期待|希望|坐等|敲碗|許願)"), "optimism"),
    (re.compile(r"一定會"), "optimism"),

    # K - Pride 自豪
    (re.compile(r"(神作|最強|第一|頂天|無敵|我的超人)"), "pride"),
    (re.compile(r"(YYDS|yyds)"), "admiration"),
    (re.compile(r"天下第一"), "pride"),

    # L - Relief 寬慰
    (re.compile(r"(還好|幸好|虛驚|沒事就好|鬆一口氣|嚇死我了[還幸])"), "relief"),

    # M - Anger 憤怒
    (re.compile(r"(氣死|可惡|該死|混蛋|去死|怒|幹|凎|操|肏)"), "anger"),

    # N - Annoyance 煩惱
    (re.compile(r"(煩死|受不了|又來|套路|膩了|拖戲|無聊)"), "annoyance"),

    # O - Disappointment 失望
    (re.compile(r"(失望|可惜|爛尾|不夠看|就這\??|就這？)"), "disappointment"),
    (re.compile(r"這[^，]*不行"), "disappointment"),

    # P - Disapproval 不認可
    (re.compile(r"(不合理|智商掉線|邏輯死去|吐槽|搞毛|三觀|好醜)"), "disapproval"),

    # Q - Disgust 厭惡
    (re.compile(r"(噁心|嘔|太油|宅味|看不下去)"), "disgust"),
    (re.compile(r"(三小|殺小|沙小|殺洨|三洨|沙洨|尛|工三小)"), "disgust"),

    # R - Embarrassment 尷尬
    (re.compile(r"(尷尬|腳趾|中二|好尬|不忍直視)"), "embarrassment"),

    # S - Fear 恐懼
    (re.compile(r"(恐怖|好可怕|嚇死|心臟受不了|毛骨悚然)"), "fear"),
    (re.compile(r"(胃痛|不敢看)"), "fear"),

    # T - Grief 悲痛
    (re.compile(r"(便當|領便當)"), "grief"),
    (re.compile(r"(太虐|虐爆|心痛|QAQ|QQ)"), "grief"),

    # U - Nervousness 緊張
    (re.compile(r"(好緊張|緊張|怕|擔心|千萬不|拜託不|挫)"), "nervousness"),

    # V - Remorse 自責
    (re.compile(r"(對不起|抱歉|我錯了|我不該|都怪我|我太廢)"), "remorse"),

    # W - Sadness 悲傷
    (re.compile(r"(哭爆|哭了|眼淚|洋蔥|催淚|鼻酸|心酸|好感人|嗚嗚)"), "sadness"),

    # X - Confusion 困惑
    (re.compile(r"(蛤\??|不懂|什麼意思|在幹嘛|WTF|wtf|何意味)"), "confusion"),
    (re.compile(r"蝦"), "confusion"),

    # Y - Curiosity 好奇
    (re.compile(r"(想知道|好奇|為什麼|求解)"), "curiosity"),

    # Z - Realization 領悟
    (re.compile(r"(原來如此|懂了|原來是|伏筆|回收|恍然大悟)"), "realization"),
    (re.compile(r"所以[^，]*是"), "realization"),

    # @A - Surprise 驚訝
    (re.compile(r"^[?？\s!！]+$"), "surprise"),
    (re.compile(r"(真假|不是吧|不會吧|太扯了|震撼|震驚|天啊)"), "surprise"),
    (re.compile(r"(臥槽|靠北|靠腰|夭壽|何止)"), "surprise"),
]


def filter_noise(text: str) -> FilterAction | None:
    if any(kw in text for kw in NOISE_KEYWORDS):
        return "noise"
    if re.match(r"^\d+[\d\/\s\:\-\.]*$", text):
        return "number"
    if not text.strip():
        return "empty"
    return None


def normalize(text: str) -> str:
    text = re.sub(r"([?？])\1+", r"\1\1", text)
    text = re.sub(r"([!！])\1+", r"\1\1", text)
    text = re.sub(r"([哈])\1{2,}", r"\1\1", text)
    text = re.sub(r"([笑])\1{2,}", r"\1\1", text)
    text = re.sub(r"(.)\1{4,}", r"\1\1\1", text)
    return text.strip()


def rule_based_match(text: str) -> RuleResult:
    for pattern, emotion in DIRECT_EMOTION_RULES:
        if pattern.search(text):
            return (text, emotion)
    return None


def preprocess(text: str) -> tuple[str | None, str | None, str | None]:
    for pattern, emotion in NUMBER_SLANG.items():
        if pattern.match(text):
            return (text, emotion, None)

    action = filter_noise(text)
    if action is not None:
        return (None, None, action)

    normalized = normalize(text)

    rule_result = rule_based_match(normalized)
    if rule_result is not None:
        return (rule_result[0], rule_result[1], None)

    return (normalized, None, None)
