# from numpy import less
from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./heart.csv")
df2 = df.drop(labels=['target'],axis=1)
X = df2.iloc[:, :]
y = df['target'].iloc[:]
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=20)

#feature scaling
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


app = Flask(__name__)

model = joblib.load("Heart_Model.pkl")

@app.route("/", methods=['GET','POST'])
def FrontPage():
    # return f"<h1>{model.predict(sc_X.transform([[45,1,3,110,264,0,1,132,0,1.2,1,0,3]]))}</h1>"
    # return f"<h1>{model.predict(sc_X.transform([[51,1,2,110,175,0,1,123,0,0.6,2,0,2]]))}</h1>"
    return render_template("index.html")

@app.route("/reply", methods=['GET','POST'])
def FrontPag():
    # Age:age, Sex:sex, Cp:cp, Bp:bp, Chol:chol, Fbs:fbs, Restecg:restecg, Thalach:thalach, Exang:exang, Oldpeak:oldpeak, Slope:slope, Ca:ca, Thal:thal
    age = request.json['Age']
    sex = request.json['Sex']
    cp = request.json['Cp']
    bp = request.json['Bp']
    chol = request.json['Chol']
    fbs = request.json['Fbs']
    restecg = request.json['Restecg']
    thalach = request.json['Thalach']
    exang = request.json['Exang']
    oldpeack = request.json['Oldpeak']
    slope = request.json['Slope']
    ca = request.json['Ca']
    thal = request.json['Thal']
    result = model.predict(sc_X.transform([[int(age),int(sex),int(cp),int(bp),int(chol),int(fbs),int(restecg),int(thalach),int(exang),float(oldpeack),int(slope),int(ca),int(thal)]]))
    print(result)
    return str(result[0])

if __name__ == "__main__":
    app.run(debug=True,port=8000)