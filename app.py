from flask import Flask, request, render_template, Response
from flask_apscheduler import APScheduler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from helper import  get_snnTorch_preds, scrape, create_figure


app = Flask(__name__, template_folder='templates')
scheduler = APScheduler()
scrape_every = 30

#hardcoded yfinance request params
stock_name = 'SBUX'
interval='1d'
start_date='2019-12-11'
end_date='2020-09-24'
pred_end_date='2020-12-10'

#plots ticker history
@app.route('/plot.png', methods=("POST", "GET"))
@scheduler.task(id = 'Plot PNG', trigger="interval", seconds=scrape_every)
def plot_png():
    df = scrape(stock_name, start_date, pred_end_date, interval)
    with app.app_context():
        print('plotted')
        pdf, loss = get_snnTorch_preds(df)
        fig = create_figure(pdf, col='forecast_vol')
        fig.savefig('ticker_forecasts.png')
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')


#plots predicted ticker prices
@app.route('/plot2.png', methods=("POST", "GET"))
@scheduler.task(id = 'Plot PNG 2', trigger="interval", seconds=scrape_every)
def plot_png2():
    df = scrape(stock_name, start_date, pred_end_date, interval)
    with app.app_context():
        data = request.form.get("req", "field: req was not provided")
        print('plotted 2')
        ndf = df.loc[:pred_end_date]
        fig = create_figure(ndf[['Volume']], col='ground_vol')
        fig.savefig('ticker_history.png')
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')


#posts ticker history as table in flask app
@app.route('/', methods=("POST", "GET"))
@scheduler.task(id = 'Scheduled Task', trigger="interval", seconds=scrape_every)
def scheduleTask():
    df = scrape(stock_name, start_date, pred_end_date, interval)
    with app.app_context():
        df['Date'] = df.index
        df.reset_index(drop=True, inplace=True)
        df = df[['Date','Open','High','Low','Close','Adj Close','Volume']].sort_values(by='Date', ascending=False)
        return render_template('simple.html', column_names=df.columns.values, row_data=list(df.values.tolist()), zip=zip)



if __name__ == "__main__":
    scheduler.init_app(app)
    scheduler.start()
    app.run()

    #https://finance.yahoo.com/chart/CL%3DF



    '''start_date =  datetime.today() - timedelta(days=100) 
    end_date = datetime.today() - timedelta(days=5)
    pred_end_date = datetime.today() + timedelta(days=1)
    start_date = start_date.strftime('%Y-%m-%d')
    end_date = end_date.strftime('%Y-%m-%d')
    pred_end_date = pred_end_date.strftime('%Y-%m-%d')'''