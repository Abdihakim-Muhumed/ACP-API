from flask import Flask
from flask_restx import Api,Resource
from flask_restx import reqparse
from .model import Model

app=Flask(__name__)
api=Api(app, version="1.0", title="ACP Tool", description="Account Prediction Tool",
          default="ACP", default_label="")

model=Model("../datasets/Train.csv","../datasets/Test.csv")
model.trainModel()

parser = reqparse.RequestParser()
parser.add_argument('country', type=str, help='country',choices=("Kenya","Uganda","Rwanda","Tanzania"))
parser.add_argument('location_type', type=str, help='location type',choices=("Urban","Rural"))
parser.add_argument('cellphone_access', type=str, help='cellphone access',choices=("No","Yes"))
parser.add_argument('year', type=int, help='year')
parser.add_argument('household_size', type=int, help='household size')
parser.add_argument('age_of_respondent', type=int, help='age of respondent')
parser.add_argument('gender_of_respondent', type=str, help='gender of respondent',choices=("Male","Female"))
parser.add_argument('relationship_with_head', type=str, help='relationship with head',choices=('Child', 'Head of Household', 'Other non-relatives', 'Other relative', 'Parent', 'Spouse'))
parser.add_argument('marital_status', type=str, help='marital status',choices=('Divorced/Seperated', 'Dont know', 'Married/Living together', 'Single/Never Married', 'Widowed'))
parser.add_argument('education_level', type=str, help='education level',choices=('No formal education', 'Other/Dont know/RTA', 'Primary education', 'Secondary education', 'Tertiary education', 'Vocational/Specialised training', 'Dont Know/Refuse to answer'))
parser.add_argument('job_type', type=str, help='job type',choices=('Farming and Fishing', 'Formally employed Government', 'Formally employed Private', 'Government Dependent', 'Informally employed', 'No Income', 'Other Income', 'Remittance Dependent', 'Self employed'))


@api.route("/predict")
class Predict(Resource):
    
    @staticmethod
    @api.doc(parser=parser)
    @api.response(200, 'success')
    def post():
        '''Function to handle the POST request.
        '''
        args=parser.parse_args()
        
        columns=[]
        row=[]
        
        for arg in args:
            val=args[arg]
            
            columns.append(arg)
            row.append(val)
        
        rowdata=[row,]
        
        prediction,accuracy=model.getPrediction(rowdata,columns)
        
        #prediction=prediction.replace("[","").replace("]","")
        
        yes=prediction[1]

        if yes==1:
            return(f"YES THIS PERSON IS LIKELY TO OPEN A NEW ACCOUNT. MODEL ACCURACY: {accuracy*100}% .") 
        else:
            return(f"NO THIS PERSON IS NOT LIKELY TO OPEN A NEW ACCOUNT. MODEL ACCURACY {accuracy*100}% .")       
        
        
if __name__=="__main__":
    app.run()