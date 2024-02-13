from src.Campus_Placement.pipelines.prediction_pipeline import CustomData,PredictPipeline

from flask import Flask,request,render_template,jsonify


app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")
    
    else:
        data=CustomData(
            
            gender=request.form.get('gender'),
            ssc_p = float(request.form.get('ssc_p')),
            ssc_b = request.form.get('ssc_b'),
            hsc_p=float(request.form.get('hsc_p')),
            hsc_b= request.form.get('hsc_b'),
            hsc_s= request.form.get('hsc_s'),
            degree_p =float(request.form.get('degree_p')),
            degree_t =request.form.get('degree_p'),
            workex =request.form.get('workex'),
            etest_p =float(request.form.get('etest_p')),
            specialisation= request.form.get('specialisation'),
            mba_p= float(request.form.get('mba_p'))
        )
        # this is my final data
        final_data=data.get_data_as_dataframe()
        
        predict_pipeline=PredictPipeline()
        
        pred=predict_pipeline.predict(final_data)
        
        result=pred[0]
        
        return render_template("result.html",final_result=result)

#execution begin
if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)