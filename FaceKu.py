import cv2
import numpy as np
import time
import pandas as pd
import os
import mysql.connector
from FaceModel import FaceModel
import random


class FaceKu():
    """人脸库主要提供人脸注册，删除，更新，人脸识别等功能
    Attributes:
        feature_list: type(list)  face features for all valid faces
        name_list: type(list)     name labels to feature list  
    Methods:
        add_face()
        del_face()
        update_face()
        face_compare()
    Dependency: 
        class FaceModel
    """
    feature_list = []
    name_list = []
    compare_level = 0.6
    face_store_path = "image/face/"

    def __init__(self):
        """load all features and names into memory """
        self.fm = FaceModel()
        self.mydb = mysql.connector.connect(
            host="127.0.0.1",
            database="face_db",
            user="root",
            password="Admin&123",
        )
        self.load_features()

    def load_features(self):
        cur = self.mydb.cursor()
        sql = """ select * from face_feature """
        cur.execute(sql)
        res = cur.fetchall()
        self.feature_list.clear()
        self.name_list.clear()
        for row in res:
            feature = np.frombuffer(row[1])
            self.feature_list.append(feature)
            self.name_list.append(row[3])
        cur.close()

    def __del__(self):
        self.mydb.close()
        print("destroy instance of class FaceKu")

    def add_face(self, image, person_info):
        """add one face into face feature table
        Args:
            image (str): absolute path of face photo file 
            person_info (dict): person info with input photo
        return
            res(bool): result 
        """
        res = False
        # detect face and abstract feature
        tmp = self.fm.face_encodings(image)
        feature = tmp['list_128d'][0]
        #feature = self.fm.face_encodings(image)[0]
        name = person_info['name']
        f_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "image\\face\\")
        _photo = f"{name}_{random.randint(1,100)}.png"
        image_path = os.path.join(f_dir, _photo)

        data_save = (feature, name, image_path)

        # add face into database
        rtn = self.save_to_db(data_save)
        # update memory data of feature_list and name list
        if rtn:
            cv2.imwrite(image_path, image)
            self.load_features()
            res = True

        return res

    def del_face(self, name):
        """ delete face from face feature table """
        res = False
        cur = self.mydb.cursor()

        # 查询并读取照片路径，删除照片 -- TBD
        sql_query = """ select * from face_feature where name=%s """
        data_query = (name,)
        cur.execute(sql_query, data_query)
        res_qs = cur.fetchall()

        if res_qs == 0:
            print("don't find record for ", name)
            res = False
        else:
            # 删除记录
            sql_del = """ DELETE FROM face_feature WHERE name LIKE %s """
            data_del = (name, )
            try:
                # execute delete operation
                cur.execute(sql_del, data_del)
                self.mydb.commit()
                # delete face photo file
                img_path = res_qs[0][4]
                if os.path.exists(img_path):
                    os.remove(img_path)
                print("successfully delete db record and face image ", name)
                res = True
            except Exception as e:
                self.mydb.rollback()
                print(e)
                res = False

        if res:
            self.load_features()
        cur.close()
        return res

    def update_face(self, image, person_info):
        """ update face feature with specified face image."""
        res = False
        # detect face and abstract feature
        tmp = self.fm.face_encodings(image)
        feature = tmp['list_128d'][0]
        #feature = self.fm.face_encodings(image)[0]
        name = person_info['name']
        f_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "image\\face\\")
        _photo = f"{name}_{random.randint(1,100)}.png"
        image_path = os.path.join(f_dir, _photo)
        data_save = (feature, name, image_path)

        # add face into database
        rtn = self.save_to_db(data_save, action=2)
        # update memory data of feature_list and name list
        if rtn:
            cv2.imwrite(image_path, image)
            self.load_features()
            res = True

        return res

    def face_compare(self, image):
        """ 输入照片，与人脸库比较，返回匹配者姓名 """
        # detect face and abstract feature
        tmp = self.fm.face_encodings(image)
        if len(tmp['list_128d']) == 0:
            return None

        feature_unkn = tmp['list_128d'][0]
        # feature_unkn = self.fm.face_encodings(image)[0]  # 如果返回为空，如何处理？

        # compare with feature list
        computed_distances_ordered, ordered_names = self.fm.compare_faces_ordered(
            self.feature_list, self.name_list, feature_unkn)
        '''
        for res in zip(computed_distances_ordered, ordered_names):
            print(f"Name:{res[1]}, Distance:{res[0]:0.4f}")
        '''

        # get name of matched person
        min_d = min(computed_distances_ordered)
        if min_d <= self.compare_level:
            min_index = computed_distances_ordered.index(min_d)
            name_predict = ordered_names[min_index]
            print("Predict name is ", name_predict, "distance is ", min_d)
        else:
            print("don't find similar person")
        del(computed_distances_ordered)
        del(ordered_names)

        res = {
            "name_predict": name_predict,
            "bbox": tmp['face_locations'][0],
        }
        return res

    def get_compare_level(self):
        return self.compare_level

    def set_compare_level(self, value):
        #  set compare_level
        if value < 1 and value > 0:
            self.compare_level = value
            res = True
        else:
            print("The input value should be among 0 and 1")
            res = False
        return res

    def add_face_from_excel(self, excel_file):
        # add faces from excel_file with batch mode.
        if not os.path.exists(excel_file):
            return False
        df_info = pd.read_excel("image/train/people_info.xlsx", sheet_name=0)
        f_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "image\\train\\")
        
        for index,row in df_info.iterrows():
            _photo = row['photo_path'].strip()
            _img_file = os.path.join(f_dir,_photo)
            if not os.path.exists(_img_file):
                print(f"The image for {row['name']} does not exist ")
                break
            image = _img_file
            # read and pre-process photo
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            width = img.shape[1]
            fix_width = 500
            ratio = fix_width/width
            fix_height = int(img.shape[0]*ratio)
            img = cv2.resize(img, (fix_width, fix_height), cv2.INTER_LINEAR)
            
            # detect face and abstract features
            known = self.fm.face_encodings(img)[0]
            # add to list
            self.feature_list.append(known)
            self.name_list.append(row['name'].strip())
            print(f"{index} completed, {row['name']} ")
            # save to db
            image_x = f"{image}"   # change \\ in path to \ 
            data_to_save = (known, row['people_id'], row['name'], image_x)
            self.save_to_db(data_to_save)          

        return None

    def save_to_db(self, data, action=1):
        """
        Save face feature and person label info into database
        Arguments:
            data: type(tuple),  ( feature, name, photo_path)
            actions:  type(int), 1 - Insert, 2 - Update
        Returns:
            res: type(boolean) 
        """
        cur = self.mydb.cursor()
        f, name, photo = data
        fs = f.tobytes()
        sql_query = """ select * from face_feature where name=%s """
        data_query = (name,)
        cur.execute(sql_query, data_query)
        res = cur.fetchall()
        if len(res) > 0:
            print(f"{data[2]} already exists, skip to insert")
            res = False
            if action == 2:
                sql_update = """" UPDATE face_feature 
                                SET feature=%s,photo_path=%s
                                WHERE name=%s """
                data_update = (fs, photo, name)
                try:
                    cur.execute(sql_update, data_update)
                    self.mydb.commit()
                    # delete face photo file
                    img_path = res[0][4]
                    if os.path.exists(img_path):
                        os.remove(img_path)
                    print("Data successfully updated for name: ", name)
                    res = True
                except Exception as e:
                    self.mydb.rollback()
                    print(e)
                    res = False
            else:
                print("该用户已存在，添加操作中止")

        else:
            sql_insert = (
                "INSERT INTO face_feature(feature, name, photo_path)"
                "VALUES (%s, %s, %s)"
            )
            data_insert = (fs, name, photo)
            print(name, photo)
            try:
                cur.execute(sql_insert, data_insert)
                self.mydb.commit()
                print("Data inserted for name: ", name)
                res = True
            except Exception as e:
                # Rolling back in case of error
                self.mydb.rollback()
                print(e)
                res = False

        # Closing the connection
        cur.close()
        return res


if __name__ == "__main__":
    fk = FaceKu()
    # 准备测试数据
    imgx = cv2.imread('image/train/t-1.jpg')
    width = imgx.shape[1]
    input_width = 500
    ratio = input_width/width
    input_height = int(imgx.shape[0]*ratio)
    imgx = cv2.resize(imgx, (input_width, input_height), cv2.INTER_LINEAR)

    people_info = {
        'name': 'test100',
        "people_id": 999
    }
    start_time = time.time()
    res = fk.face_compare(imgx)
    end_time = time.time()
    t_diff = end_time - start_time
    print(f"total time: { t_diff} ")
    print("识别结果: ", res)

    #res = fk.add_face(imgx, people_info)
    #print("人员添加结果: ", res)

    time.sleep(2)
   # res = fk.del_face(people_info['name'])
   # print("人员删除结果: ", res)
