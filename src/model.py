#!/usr/bin/python3
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import copy


class Model:
    def __init__(self,trainDf=None,testDf=None):   
        if trainDf and testDf:
            self.datafile=trainDf
            self.testfile=testDf
 
    def doEncode(self,df,col):
        '''Function to encode data into binary values.
        '''
        dummies=pd.get_dummies(df[col])
        df.drop(columns=col,inplace=True)
        df=pd.concat([df,dummies],axis=1)
        return df
            
    def trainModel(self):
        ''' Function to train and test the model using the provided dataset.
        '''
        self.train_data=pd.read_csv(self.datafile)
        self.test_data=pd.read_csv(self.testfile)
        self.targetCol=["country","location_type","cellphone_access","gender_of_respondent","relationship_with_head","marital_status","education_level","job_type"]
        
        self.input_data=self.train_data.drop(columns="bank_account")

        self.output_data=self.train_data["bank_account"]
        self.output_data=pd.DataFrame(self.output_data)
        
        self.input_data.drop(columns="uniqueid",inplace=True)

        self.input_data=self.doEncode(self.input_data,"country")
        self.input_data=self.doEncode(self.input_data,"location_type")
        self.input_data=self.doEncode(self.input_data,"cellphone_access")
        self.input_data=self.doEncode(self.input_data,"gender_of_respondent")
        self.input_data=self.doEncode(self.input_data,"relationship_with_head")
        self.input_data=self.doEncode(self.input_data,"marital_status")
        self.input_data=self.doEncode(self.input_data,"education_level")
        self.input_data=self.doEncode(self.input_data,"job_type")
        
        self.columns=self.input_data.columns.values.tolist()

        self.output_data=self.doEncode(self.output_data,"bank_account")
        
        


        self.model=DecisionTreeClassifier(criterion = 'entropy', random_state = 42)

        x_train,x_test,y_train,y_test=train_test_split(self.input_data,self.output_data,test_size=0.2)
        self.model.fit(x_train,y_train)

        predictions=self.model.predict(x_test)
        self.score=accuracy_score(y_test,predictions)
        
        
    def getPrediction(self,rowdata=None,columns=None):
        '''Function to get prdictions for the passed data and columns.
        '''
        if not rowdata or not columns:
            return False
        
        
        input_data=pd.DataFrame(rowdata,columns=columns)
        
        input_data=self.doEncode(input_data,"country")
        input_data=self.doEncode(input_data,"location_type")
        input_data=self.doEncode(input_data,"cellphone_access")
        input_data=self.doEncode(input_data,"gender_of_respondent")
        input_data=self.doEncode(input_data,"relationship_with_head")
        input_data=self.doEncode(input_data,"marital_status")
        input_data=self.doEncode(input_data,"education_level")
        input_data=self.doEncode(input_data,"job_type")
        
        
                
        predata=[]       
        for col in self.columns:
            if col in input_data.columns.values.tolist():
                predata.append(1)
            else:
                predata.append(0)
            
        
        data=[predata,]
        
        final_data=pd.DataFrame(data,columns=self.columns)
                
        prediction=self.model.predict(final_data)
                
        return prediction[0],self.score