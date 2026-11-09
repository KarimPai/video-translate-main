"""Official-name reference for selecting bounded Hololive speech hints.

The catalog is *not* injected wholesale into the Whisper prompt. Names are
selected only from a stream's title/channel metadata, or manually supplied
hotwords in config.toml. Avoid guessing who is speaking from the video ID.
Reference: https://hololive.hololivepro.com/en/talents
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Talent:
    official: str
    japanese: str
    hint: str

# Officially listed active, affiliate, and alumni Hololive talents as of
# November 2026. Japanese names make title/channel matching practical.
# Hint is short and pronounceable; its purpose is spelling, not identity.
TALENTS: tuple[Talent, ...] = (
    Talent("Tokino Sora", "ときのそら", "Sora"),
    Talent("Robocosan", "ロボ子さん", "Roboco"),
    Talent("Aki Rosenthal", "アキ・ローゼンタール", "Aki Rosenthal"),
    Talent("Akai Haato", "赤井はあと", "Haato"),
    Talent("Shirakami Fubuki", "白上フブキ", "Fubuki"),
    Talent("Natsuiro Matsuri", "夏色まつり", "Matsuri"),
    Talent("Nakiri Ayame", "百鬼あやめ", "Ayame"),
    Talent("Yuzuki Choco", "癒月ちょこ", "Choco"),
    Talent("Oozora Subaru", "大空スバル", "Subaru"),
    Talent("AZKi", "AZKi", "AZKi"),
    Talent("Ookami Mio", "大神ミオ", "Mio"),
    Talent("Sakura Miko", "さくらみこ", "Miko"),
    Talent("Nekomata Okayu", "猫又おかゆ", "Okayu"),
    Talent("Inugami Korone", "戌神ころね", "Korone"),
    Talent("Hoshimachi Suisei", "星街すいせい", "Suisei"),
    Talent("Usada Pekora", "兎田ぺこら", "Pekora"),
    Talent("Shiranui Flare", "不知火フレア", "Flare"),
    Talent("Shirogane Noel", "白銀ノエル", "Noel"),
    Talent("Houshou Marine", "宝鐘マリン", "Marine"),
    Talent("Tsunomaki Watame", "角巻わため", "Watame"),
    Talent("Tokoyami Towa", "常闇トワ", "Towa"),
    Talent("Himemori Luna", "姫森ルーナ", "Luna"),
    Talent("Yukihana Lamy", "雪花ラミィ", "Lamy"),
    Talent("Momosuzu Nene", "桃鈴ねね", "Nene"),
    Talent("Shishiro Botan", "獅白ぼたん", "Botan"),
    Talent("Omaru Polka", "尾丸ポルカ", "Polka"),
    Talent("La+ Darknesss", "ラプラス・ダークネス", "Laplus"),
    Talent("Takane Lui", "鷹嶺ルイ", "Lui"),
    Talent("Hakui Koyori", "博衣こより", "Koyori"),
    Talent("Kazama Iroha", "風真いろは", "Iroha"),
    Talent("Sakamata Chloe", "沙花叉クロヱ", "Chloe"),
    Talent("Ayunda Risu", "アユンダ・リス", "Risu"),
    Talent("Moona Hoshinova", "ムーナ・ホシノヴァ", "Moona"),
    Talent("Airani Iofifteen", "アイラニ・イオフィフティーン", "Iofi"),
    Talent("Kureiji Ollie", "クレイジー・オリー", "Ollie"),
    Talent("Anya Melfissa", "アーニャ・メルフィッサ", "Anya"),
    Talent("Pavolia Reine", "パヴォリア・レイネ", "Reine"),
    Talent("Vestia Zeta", "ベスティア・ゼータ", "Zeta"),
    Talent("Kaela Kovalskia", "カエラ・コヴァルスキア", "Kaela"),
    Talent("Kobo Kanaeru", "こぼ・かなえる", "Kobo"),
    Talent("Mori Calliope", "森カリオペ", "Calliope"),
    Talent("Takanashi Kiara", "小鳥遊キアラ", "Kiara"),
    Talent("Ninomae Ina'nis", "一伊那尓栖", "Ina'nis"),
    Talent("Watson Amelia", "ワトソン・アメリア", "Amelia"),
    Talent("IRyS", "IRyS", "IRyS"),
    Talent("Ouro Kronii", "オーロ・クロニー", "Kronii"),
    Talent("Hakos Baelz", "ハコス・ベールズ", "Baelz"),
    Talent("Shiori Novella", "シオリ・ノヴェラ", "Shiori"),
    Talent("Koseki Bijou", "古石ビジュー", "Bijou"),
    Talent("Nerissa Ravencroft", "ネリッサ・レイヴンクロフト", "Nerissa"),
    Talent("Fuwawa Abyssgard", "フワワ・アビスガード", "Fuwawa"),
    Talent("Mococo Abyssgard", "モココ・アビスガード", "Mococo"),
    Talent("Elizabeth Rose Bloodflame", "エリザベス・ローズ・ブラッドフレイム", "Elizabeth"),
    Talent("Gigi Murin", "ジジ・ムリン", "Gigi"),
    Talent("Cecilia Immergreen", "セシリア・イマーグリーン", "Cecilia"),
    Talent("Raora Panthera", "ラオーラ・パンテーラ", "Raora"),
    Talent("Otonose Kanade", "音乃瀬奏", "Kanade"),
    Talent("Ichijou Ririka", "一条莉々華", "Ririka"),
    Talent("Juufuutei Raden", "儒烏風亭らでん", "Raden"),
    Talent("Todoroki Hajime", "轟はじめ", "Hajime"),
    Talent("Isaki Riona", "響咲リオナ", "Riona"),
    Talent("Koganei Niko", "虎金妃笑虎", "Niko"),
    Talent("Mizumiya Su", "水宮枢", "Mizumiya"),
    Talent("Rindo Chihaya", "輪堂千速", "Chihaya"),
    Talent("Kikirara Vivi", "綺々羅々ヴィヴィ", "Vivi"),
    Talent("Achichi Mela", "熱千めら", "Mela"),
    Talent("Sorashina Sopia", "宙科そぴあ", "Sopia"),
    Talent("Suzuna Tsuzuri", "鈴鳴つづり", "Tsuzuri"),
    Talent("Hyakuto Kyoko", "百灯キョーコ", "Kyoko"),
    Talent("Izuki Michiru", "井月みちる", "Michiru"),
    Talent("Hanazono Sayaka", "花園さやか", "Sayaka"),
    Talent("Kazeshiro Yuki", "風白ゆき", "Kazeshiro"),
    Talent("Minato Aqua", "湊あくあ", "Aqua"),
    Talent("Murasaki Shion", "紫咲シオン", "Shion"),
    Talent("Amane Kanata", "天音かなた", "Kanata"),
    Talent("Kiryu Coco", "桐生ココ", "Coco"),
    Talent("Gawr Gura", "がうる・ぐら", "Gura"),
    Talent("Tsukumo Sana", "九十九佐命", "Sana"),
    Talent("Ceres Fauna", "セレス・ファウナ", "Fauna"),
    Talent("Nanashi Mumei", "七詩ムメイ", "Mumei"),
    Talent("Hiodoshi Ao", "火威青", "Ao"),
)

# A subset of personal-name nicknames are distinctive enough to recognize
# from a *video title*, never from the channel title alone. Short/common
# terms such as "Mio", "Aki", "Luna", "Marine" can have unrelated meanings.
_DISTINCTIVE_NICKNAMES = frozenset({
    "watame", "okayu", "korone", "pekora", "fubuki", "suisei",
    "subaru", "ayame", "botan", "koyori", "kanade",
    "matsuri", "chihaya", "chloe", "raden", "mumei", "kronii",
    "kobo", "gura", "towa", "shion", "lamy", "iroha",
})


def _match_full(text: str, talent: Talent) -> bool:
    if not text:
        return False
    name = talent.official
    variants = [name]
    # Explicit romanized full names and official Japanese spellings.
    if talent.japanese:
        variants.append(talent.japanese)
    # The punctuation in La+ Darknesss and Ina'nis can differ in metadata.
    for variant in variants:
        if re.search(r"[A-Za-z]", variant):
            term = re.escape(variant)
            if re.search(r"(?<![A-Za-z])" + term + r"(?![A-Za-z])", text, re.I):
                return True
        elif variant in text:
            return True
    return False


def _match_nickname(title: str, talent: Talent) -> bool:
    key = talent.hint.casefold()
    if key not in _DISTINCTIVE_NICKNAMES:
        return False
    if re.search(r"(?<![A-Za-z])" + re.escape(talent.hint) +
                 r"(?![A-Za-z])", title, re.I):
        return True
    # Japanese short-name titles: tag #わため, etc.
    jp = re.sub(r"^[^ぁ-ゟァ-ヿ]+", "", talent.japanese)
    return len(jp) >= 3 and jp in title


def select_hololive_hints(
    metadata: dict | None,
    *,
    enabled: bool = True,
    limit: int = 6,
) -> list[str]:
    """Select up to six names grounded by channel/title metadata.

    No title/channel -> no auto hints. Manual config hotwords are separate.
    Full channel matches outrank any title/nickname matches.
    """
    if not enabled or not isinstance(metadata, dict) or limit <= 0:
        return []
    channel = str(metadata.get("channel") or "")[:300]
    title = str(metadata.get("title") or "")[:500]
    selected: list[tuple[int, int, Talent]] = []
    for order, talent in enumerate(TALENTS):
        priority = (
            0 if _match_full(channel, talent) else
            1 if _match_full(title, talent) else
            2 if _match_nickname(title, talent) else None
        )
        if priority is not None:
            selected.append((priority, order, talent))
    selected.sort(key=lambda t: (t[0], t[1]))
    found: list[str] = []
    seen = set()
    for _, _, talent in selected:
        name = talent.hint
        if name.casefold() not in seen:
            seen.add(name.casefold())
            found.append(name)
        if len(found) >= max(0, min(6, limit)):
            break
    return found
