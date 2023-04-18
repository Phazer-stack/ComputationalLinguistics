import json
import requests
import re
import csv
import pandas as pd
import xmltodict as xmltodict


class OireachtasCorpus:
    def __init__(self, tdMemberID, startDate, endDate, limit):
        self.homeURL = "https://api.oireachtas.ie/v1"
        self.td_ID = "https://data.oireachtas.ie/ie/oireachtas/member/id/" + tdMemberID
        self.startDate = startDate
        self.endDate = endDate
        self.limit = limit
        self.debateDate2XLMDic = self.scrapeDebates()
        self.tsv = "TD_TSV"
        self.debates2TSV(self.tsv)

    def scrapeDebates(self):

        debatesURL = self.homeURL + "/debates"

        payload = {"date_start": "2018-01-01", "end_date": "2099-01-01", "limit": 4000, "member_id": self.td_ID}
        res = requests.get(debatesURL, verify=False, params=payload)

        if res.status_code == 200:
            print(res.status_code.__str__() + ", access to debates oireachtas URL success")
            debatesJSONObject = res.json()
            new_json = json.dumps(debatesJSONObject, indent=2)
            print(new_json)

            debateDic = debatesJSONObject["results"]
            print("debates stored")

            debateDate2XMLDic = {}

            for debateResult in debateDic:
                debateRecordDic = debateResult['debateRecord']
                debateDate = debateRecordDic['date']
                forumType = debateRecordDic['debateType']
                dicContent = debateRecordDic['formats']
                xmlURIDic = dicContent['xml']
                xmlURI = xmlURIDic['uri']
                if debateDate in debateDate2XMLDic:
                    if xmlURI not in debateDate2XMLDic[debateDate]:
                        debateDate2XMLDic[debateDate].append(xmlURI)
                else:
                    debateDate2XMLDic[debateDate] = [xmlURI]

            print("debate sections represented by dates")
            return debateDate2XMLDic

    def debates2TSV(self, file):

        with open(file, 'w', encoding="utf-8") as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['name', 'party', 'age', 'constituency', 'forum', 'd.o.c', 'topic', 'ordinal', 'text'])
            ordinal = 0

            for dateKey in self.debateDate2XLMDic:
                date = dateKey
                dateList = dateKey.split("-")
                age = int(dateList[0]) - 1969
                for debateXMLURI in self.debateDate2XLMDic[dateKey]:
                    xmlDic = self.xmlURI2Dic(debateXMLURI)
                    td_forum_text_list = self.xmlDic2TdText(xmlDic)
                    for contribution in td_forum_text_list:
                        (ordinal,forum,text) = contribution
                        tsv_writer.writerow(["Mary Lou McDonald", "Sinn Féin", age, "Dublin Central", forum, date, "---", ordinal, text])
        return out_file


    def xmlURI2Dic(self, uri):
        text_xmluri = uri
        res_xml = requests.get(text_xmluri, verify=False)
        if res_xml.status_code == 200:
            print(res_xml.status_code.__str__() + ", access to debates text XML oireachtas URL success")
            xml_data = xmltodict.parse(res_xml.content)
            return xml_data
        else:
              print("request failed")


    def xmlDic2TdText(self, xmlDic):
        content_list = []
        ordinal = 0
        contentText = ""
        homeDebateDic = xmlDic["akomaNtoso"]
        mainDebateDic = homeDebateDic["debate"]
        debateBodyDic = mainDebateDic["debateBody"]
        for key in debateBodyDic:
            for section in debateBodyDic[key]:
                if (section['@name'] == "debate" or section['@name'] ==  "questions") == True:
                    if 'speech' in section:
                        speechDic = section['speech']
                        if type(speechDic) is list:
                            for speech in speechDic:
                                if speech['@by'] == "#MaryLouMcDonald":
                                    if type(speech['p']) is list:
                                        for paragraph in speech['p']:
                                            contentText = contentText + paragraph['#text']
                                    elif type(speech['p']) is dict:
                                        paragraph = speech['p']
                                        contentText = contentText + paragraph['#text']
                        elif type(speechDic) is dict:
                            if speechDic['@by'] == "#MaryLouMcDonald":
                                for paragraph in speechDic['p']:
                                    contentText = contentText + paragraph['#text']

                if (section['@name'] == "debate") == True and contentText != "":
                    ordinal = ordinal + 1
                    my_triple = (ordinal, "debate", contentText)
                    contentText = ""
                    content_list.append(my_triple)

                elif(section['@name'] == "questions") == True and contentText != "":
                    ordinal = ordinal + 1
                    my_triple = (ordinal, "question", contentText)
                    contentText = ""
                    content_list.append(my_triple)


        return content_list
