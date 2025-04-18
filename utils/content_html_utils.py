import json
import re
import logging
import jionlp
import urllib
from typing import List,Dict
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def clean_tuwen_article(content: str) -> str:
    if '"rich_content"' in content:
        return clean_weitoutiao_article(content)
    return jionlp.clean_html(_remove_span_tags(content))[0]

def clean_weitoutiao_article(content: str) -> str:
    result = content
    try:
        result = json.loads(content)["rich_content"]["text"]
    except Exception as e:
        result = re.search(r"{\"rich_content\":{\"text\":\".*?\",\"spans", content, re.DOTALL)[0].replace('{"rich_content":{"text":"', "").replace('","spans', "")
    return result

def _remove_span_tags(text):
    clean_text = re.sub('<span class="sub-title".*?>.*?</span>', '', text, flags=re.DOTALL)
    clean_text = re.sub('<a class="author-name-link pgc-link".*?>.*?</a>', '', clean_text, flags=re.DOTALL)
    clean_text = re.sub('<div class="tt-title".*?>.*?</div>', '', clean_text, flags=re.DOTALL)
    clean_text = re.sub('{!-- PGC_.*?--}', '', clean_text, flags=re.DOTALL)
    return clean_text

if __name__ == '__main__':
    test_str0 = """<html><body><p>老鼠吃转基因会得肿瘤？谣言！</p><p>推广转基因的都是内鬼？谣言！</p><p>转基因影响子孙后代？谣言！</p><p>转基因对人体有害？谣言！</p><img src="{{image_domain}}{"attribute": "{\"style\": \"-webkit-tap-highlight-color: transparent;outline: 0px;width: 676.989px !important;visibility: visible !important;\", \"data-galleryid\": \"\", \"data-imgfileid\": \"502804750\", \"data-backh\": \"2918\", \"data-type\": \"jpeg\", \"data-backw\": \"578\", \"data-s\": \"300,640\", \"data-ratio\": \"5.05\", \"alt\": \"\\u56fe\\u7247\", \"class\": \"rich_pages wxw-img\", \"data-w\": \"640\"}", "fingerprint": 653085423393132791, "hash_id": 680767906309392690, "height": 4994, "id": 3656495065626013900, "image_type": 1, "md5": "097292cdce0ec9325dc61885d001af30", "mimetype": "image/webp", "near_dup_id": "17747053411643719273", "uri": "tos-cn-i-tjoges91tu/5738a024a0fdad7cbd4a7480bc2bb6ed", "url_id": 2380196657794651540, "web_uri": "tos-cn-i-tjoges91tu/5738a024a0fdad7cbd4a7480bc2bb6ed", "width": 989}"><img src="{{image_domain}}{"attribute": "{\"style\": \"-webkit-tap-highlight-color: transparent;outline: 0px;width: 676.989px !important;visibility: visible !important;\", \"data-galleryid\": \"\", \"data-imgfileid\": \"502804751\", \"data-backh\": \"3012\", \"data-type\": \"jpeg\", \"data-backw\": \"578\", \"data-s\": \"300,640\", \"data-ratio\": \"5.2125\", \"alt\": \"\\u56fe\\u7247\", \"class\": \"rich_pages wxw-img\", \"data-w\": \"640\"}", "fingerprint": 3102717880732832085, "hash_id": 18122796732268681470, "height": 5155, "id": 8353840094501749843, "image_type": 1, "md5": "fb811b9bfb7c18fee077703eede1766f", "mimetype": "image/webp", "near_dup_id": "11581881368714937466", "uri": "tos-cn-i-tjoges91tu/f534b43a572a23d4a60b41bf00602e8d", "url_id": 907116963961819403, "web_uri": "tos-cn-i-tjoges91tu/f534b43a572a23d4a60b41bf00602e8d", "width": 989}"><img src="{{image_domain}}{"attribute": "{\"style\": \"-webkit-tap-highlight-color: transparent;outline: 0px;width: 676.989px !important;visibility: visible !important;\", \"data-galleryid\": \"\", \"data-imgfileid\": \"502804752\", \"data-backh\": \"3161\", \"data-type\": \"jpeg\", \"data-backw\": \"578\", \"data-s\": \"300,640\", \"data-ratio\": \"5.4703125\", \"alt\": \"\\u56fe\\u7247\", \"class\": \"rich_pages wxw-img\", \"data-w\": \"640\"}", "fingerprint": 3258710313519037653, "hash_id": 10583686442566326592, "height": 5410, "id": 1323002196333828591, "image_type": 1, "md5": "92e0cebea7489940676485c29f85eb30", "mimetype": "image/webp", "near_dup_id": "9636954453215483030", "uri": "tos-cn-i-tjoges91tu/dc42242ff2ff33c58b167cca37c3f470", "url_id": 14189739540462357187, "web_uri": "tos-cn-i-tjoges91tu/dc42242ff2ff33c58b167cca37c3f470", "width": 989}"><img src="{{image_domain}}{"attribute": "{\"style\": \"-webkit-tap-highlight-color: transparent;outline: 0px;width: 676.989px !important;visibility: visible !important;\", \"data-galleryid\": \"\", \"data-imgfileid\": \"502804749\", \"data-backh\": \"3162\", \"data-type\": \"jpeg\", \"data-backw\": \"578\", \"data-s\": \"300,640\", \"data-ratio\": \"5.471875\", \"alt\": \"\\u56fe\\u7247\", \"class\": \"rich_pages wxw-img\", \"data-w\": \"640\"}", "fingerprint": 1443878619642680533, "hash_id": 10626750709598335913, "height": 5411, "id": 9066082583012848572, "image_type": 1, "md5": "9379cd77aa3d33a9055a671c87a6c224", "mimetype": "image/webp", "near_dup_id": "4534761051977140219", "uri": "tos-cn-i-tjoges91tu/41b6ad792051fe3fb3d39f4282ef558f", "url_id": 9036067044333074777, "web_uri": "tos-cn-i-tjoges91tu/41b6ad792051fe3fb3d39f4282ef558f", "width": 989}"><p class="pgc-img-caption">来源：农财网种业宝典</p></body></html>"""
    test_str1 = """<img src="{{image_domain}}{"fingerprint": 10436364479122794457, "hash_id": 4459255574208750464, "height": 1088, "id": 18423259679237501616, "image_type": 1, "md5": "3de275518f64a380d2d7fcb4c9dbab99", "mimetype": "image/jpeg", "near_dup_id": 0, "uri": "tos-cn-i-tjoges91tu/cd82f0c1b24a40197c4bb794b4d389a6", "url_id": 13014493153196742312, "web_uri": "tos-cn-i-tjoges91tu/cd82f0c1b24a40197c4bb794b4d389a6", "width": 1700}"><p>直播吧4月16日讯 今天，森林狼主帅芬奇在接受采访时谈到了爱德华兹。</p><p>谈及球队在让爱德华兹保持参与度方面取得了怎样的进步时，芬奇表示：“我觉得这得从爱德华兹自身说起。他对场上局势的判断越来越敏锐了。当爱德华兹和兰德尔都有冲击三双的表现时，我们的球队表现就会更好。”</p>"""
    test_str2 = """<p>{!-- PGC_VIDEO:{"duration":12.419,"external_covers":[],"file_sign":"82d0478e4a8c205649fc9a810e939a0a","md5":"82d0478e4a8c205649fc9a810e939a0a","new_video_url":"https://videocloud-tp.kksmg.com/pub/2025/04/16/f9f7cead-1aa9-11f0-99a4-e381e28b6b61_df28262a-efcf-37dc-90aa-e9d19c1a4d0a.mp4","sp":"toutiao","status":0,"thumb_height":1920,"thumb_url":"tos-cn-i-tjoges91tu/99f57dd6d86e0fb0bb3c90c2cbcbedb0","thumb_width":1082,"transcode_cb_time":"1744798882","transcode_status":"3","uri":"tos-cn-i-tjoges91tu/99f57dd6d86e0fb0bb3c90c2cbcbedb0","vid":"v0201fg10000cvvo937og65il51l6t0g","video_cover_url":"https://p.statickksmg.com/cont/2025/04/16/f9f7cead-1aa9-11f0-99a4-e381e28b6b61_5b61d8b8-1004-3312-a667-b4ecc421697d.jpg","video_size":{"1080p":{"duration":12.419,"file_size":2997497,"h":1920,"w":1080},"high":{"duration":12.419,"file_size":871381,"h":854,"w":480},"normal":{"duration":12.419,"file_size":641159,"h":640,"w":360},"ultra":{"duration":12.419,"file_size":1614881,"h":1280,"w":720}},"video_url_type":1,"vname":"TVB艺人张彦博两年多无戏拍 因家人健康问题坦言有压力","vu":"v0201fg10000cvvo937og65il51l6t0g"} --}</p><p>因曾在《金宵大厦》中饰演林哥仔人气急升的张彦博，在沉寂两年后近日携新剧重回TVB荧幕。久违地在公众面前露面，他在采访中表示自己两年多没有拍戏，虽然收入有所减少但影响有限，他还透露自己花了超100万追求音乐梦。</p><img src="{{image_domain}}{"fingerprint": 15175909364308069348, "hash_id": 3892934451659472995, "height": 1172, "id": 14294280230690885045, "image_type": 1, "md5": "36067b42150b006355bb3da6e68e0ae9", "mimetype": "image/png", "near_dup_id": "4201573386273525886", "uri": "tos-cn-i-tjoges91tu/db276e1c32e484cca361a02688b88e4f", "url_id": 10772775023641275456, "web_uri": "tos-cn-i-tjoges91tu/db276e1c32e484cca361a02688b88e4f", "width": 1084}"><p>因为近年家人身体出现问题，经济上多多少少有点压力，但会在能力范围内和家人好好享受现在的生活。</p><p>编辑: 斯雯</p><p>责编: 刘佳</p>"""
    test_str3 = """{"rich_content":{"text":"【拾金不昧】#暖新闻#\n近日，在江西省万安县宝山乡的街头，三个扎着马尾辫的小小身影，如春日暖阳般温暖人心。\n曾子涵、魏怡轩、曾子溪是宝山中心小学的学生。她们路过一家蛋糕店时，眼尖地发现地上躺着一个钱包。面对金钱的诱惑，三人坚定地选择在原地等待失主；许久没见失主前来，他们决定把钱包交给警察叔叔，以实际行动生动诠释了拾金不昧的传统美德。\n最终，民警带领三人在蛋糕店附近找到了失主，接过失而复得的钱包后，失主郑重地向民警鞠了一躬并向三名学生致谢。童谣里的“我在马路边捡到一分钱”，在新时代有了更多生动注脚。\n这也是最动人的成长仪式：拾起的是责任，交付的是担当。我们看见：传统美德从未老去，它已深植于孩子们心间。\n（本期点评 焦以璇）\n《中国教育报》2025年04月15日 第03版 版名：新闻·人物\n作者：本期点评 焦以璇","spans":[{"start":6,"length":5,"link":"sslocal://concern?cid=1620353681246212\u0026tab_sname=thread","type":2,"text":"#暖新闻#","id":"1620353681246212","id_type":8,"images":null,"extra":{"id_str":"1620353681246212"}}]},"images":[{"web_uri":"tos-cn-i-ezhpy3drpa/5195c199e9fb4e7ab9feafc167c750b7","width":1080,"height":810,"image_type":1,"mimetype":"jpeg","encrypt_web_uri":null,"secret_key":null,"encrypt_algorithm":null,"extra":{"format":"jpeg","size":"181137"}}]}"""
    
    res_0 = clean_tuwen_article(test_str0) 
    res_1 = clean_tuwen_article(test_str1)
    res_2 = clean_tuwen_article(test_str2)
    res_3 = clean_tuwen_article(test_str3)

    logger.info(f"test0:{res_0}")
    logger.info(f"test1:{res_1}")
    logger.info(f"test2:{res_2}")
    logger.info(f"test3:{res_3}")
