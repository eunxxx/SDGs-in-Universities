import argparse
import pandas as pd
import numpy as np
from pykospacing import Spacing

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--load_path",
        required=True,
        help="Path to load data file."
    )
    p.add_argument(
        "--save_path",
        required=True,
        help="Path to save preprocessed dataset."
    )

    config = p.parse_args()

    return config

def change_categories(config):
    data = pd.read_csv(config.load_path)

    conditionlist = [
        (data['분류'] == 1) | (data['분류'] == 2) | (data['분류'] == 3) | (data['분류'] == 4) | (data['분류'] == 5),
        (data['분류'] == 6) | (data['분류'] == 12) | (data['분류'] == 13) | (data['분류'] == 14) | (data['분류'] == 15),
        (data['분류'] == 7) | (data['분류'] == 8) | (data['분류'] == 9) | (data['분류'] == 10) | (data['분류'] == 11),
        (data['분류'] == 16),
        (data['분류'] == 17)
    ]

    choicelist = [1, 2, 3, 4, 5]

    data['5분류'] = np.select(conditionlist, choicelist)
    data["분류"] = data["분류"].astype('category')
    data["5분류"] = data["5분류"].astype('category')

    # 국가 용어 통일
    data.loc[data['국가'] == "Canada", "국가"] = "캐나다"
    data.loc[data['국가'] == "South Korea", "국가"] = "한국"
    data.loc[data['국가'] == "남아프리카공화국", "국가"] = "남아프리카 공화국"
    # "호주"로 용어 통일
    data.loc[data['국가'] == "Australia", "국가"] = "호주"
    data.loc[data['국가'] == "오스트레일리아", "국가"] = "호주"
    data.loc[data['국가'] == "오스트레일리아\tNewcastle australia", "국가"] = "호주"

    # 대학교명 용어 통일
    data.loc[data['대학교명'] == "Bologna University", "대학교명"] = "Bologna"
    data.loc[data['대학교명'] == "Arizona State University", "대학교명"] = "Arizona State"
    data.loc[data['대학교명'] == "Queen's University", "대학교명"] = "Queen's"
    data.loc[data['대학교명'] == "Newcastle australia", "대학교명"] = "Newcastle"
    data.loc[data['대학교명'] == "University of Glasgow", "대학교명"] = "Glasgow"
    data.loc[data['대학교명'] == "York University", "대학교명"] = "York"
    data.loc[data['대학교명'] == "Concordia University", "대학교명"] = "Concordia"
    data.loc[data['대학교명'] == "RMIT University", "대학교명"] = "RMIT"
    data.loc[data['대학교명'] == "Charles Sturt University", "대학교명"] = "Charles Sturt"
    data.loc[data['대학교명'] == "University of British Columbia", "대학교명"] = "British Columbia"
    data.loc[data['대학교명'] == "WESTERN SYDNEY UNIVERSITY", "대학교명"] = "Western Sydney"
    data.loc[data['대학교명'] == "Manchester University", "대학교명"] = "Manchester"
    data.loc[data['대학교명'] == "Bournemouth University", "대학교명"] = "Bournemouth"
    data.loc[data['대학교명'] == "Auckland universitiy", "대학교명"] = "Auckland"
    data.loc[data['대학교명'] == "University of Cantebury", "대학교명"] = "Cantebury"
    data.loc[data['대학교명'] == "Oxford University", "대학교명"] = "Oxford"
    data.loc[data['대학교명'] == "University of California", "대학교명"] = "California"
    data.loc[data['대학교명'] == "University\xa0of\xa0Wollongong", "대학교명"] = "Wollongong"
    data.loc[data['대학교명'] == "University of Technology Sydney", "대학교명"] = "Technology Sydney"
    data.loc[data['대학교명'] == "Chulalongkorn University", "대학교명"] = "Chulalongkorn"
    data.loc[data['대학교명'] == "McMaster University", "대학교명"] = "McMaster"
    data.loc[data['대학교명'] == "Sunway University", "대학교명"] = "Sunway"
    data.loc[data['대학교명'] == "University of Sydney", "대학교명"] = "Sydney"
    data.loc[data['대학교명'] == "University of Essex", "대학교명"] = "Essex"
    data.loc[data['대학교명'] == "National University of Sciences & Technology", "대학교명"] = "NUST"
    data.loc[data['대학교명'] == "Lynn University", "대학교명"] = "Lynn"
    data.loc[data['대학교명'] == "Sophia University", "대학교명"] = "Sophia"
    data.loc[data['대학교명'] == "Western University", "대학교명"] = "Western"
    data.loc[data['대학교명'] == "University of Northampton", "대학교명"] = "Northampton"
    data.loc[data['대학교명'] == "Victoria University", "대학교명"] = "Victoria"
    data.loc[data['대학교명'] == "Carnegie Mellon University", "대학교명"] = "Carnegie Mellon"
    data.loc[data['대학교명'] == "The University of Texas at Dallas", "대학교명"] = "Texas at Dallas"
    data.loc[data['대학교명'] == "University of Exeter", "대학교명"] = "Exeter"
    data.loc[data['대학교명'] == "IPB University", "대학교명"] = "IPB"
    data.loc[data['대학교명'] == "University of Manitoba", "대학교명"] = "Manitoba"
    data.loc[data['대학교명'] == "University of Hong Kong", "대학교명"] = "Hong Kong"
    data.loc[data['대학교명'] == "London School of Economics and Political Science", "대학교명"] = "LSE"
    data.loc[data['대학교명'] == "University of Eastern Finland", "대학교명"] = "Eastern Finland"
    data.loc[data['대학교명'] == "Carleton University", "대학교명"] = "Carleton"
    data.loc[data['대학교명'] == "University of Cape Town", "대학교명"] = "Cape Town"
    data.loc[data['대학교명'] == "Bahrain University", "대학교명"] = "Bahrain"
    data.loc[data['대학교명'] == "Aalborg University", "대학교명"] = "Aalborg"
    data.loc[data['대학교명'] == "UC Davis University", "대학교명"] = "UC Davis"
    data.loc[data['대학교명'] == "Ashoka University", "대학교명"] = "Ashoka"
    data.loc[data['대학교명'] == "Brock University", "대학교명"] = "Brock"
    data.loc[data['대학교명'] == "The University of Sheffield", "대학교명"] = "Sheffield"
    data.loc[data['대학교명'] == "The Chinese University of Hong Kong", "대학교명"] = "CUHK"
    data.loc[data['대학교명'] == "University of East Anglia", "대학교명"] = "East Anglia"
    data.loc[data['대학교명'] == "Tunghai University", "대학교명"] = "Tunghai"

    data.loc[data['대학교명'] == "Kyungpook National University", "대학교명"] = "Kyungpook"
    data.loc[data['대학교명'] == "경북대학교", "대학교명"] = "Kyungpook"
    data.loc[data['대학교명'] == "서울대학교", "대학교명"] = "Seoul"
    data.loc[data['대학교명'] == "아주대학교", "대학교명"] = "Ajou"
    data.loc[data['대학교명'] == "강원대학교", "대학교명"] = "Kangwon"

def spacing(config):
    data = pd.read_csv(config.load_path)

    new_sent_list = []
    for s in data["번역_전처리"]:
        new_sent = s.replace(" ", '')
        new_sent_list.append(new_sent)
    
    new_sent_list2 = []
    for s in new_sent_list:
        s = s.replace("x000D", "")
        s = s.replace("_x000D_", "")
        new_sent_list2.append(s)

    spacing = Spacing()
    kos_sent_list=[]

    for s in new_sent_list2:
        spacing = Spacing()
        kospacing_sent = spacing(s)
        kos_sent_list.append(kospacing_sent)
    
    data.to_csv(config.save_path)

if __name__ == '__main__':
    config = define_argparser()
    change_categories(config)
    spacing(config)