from flask import Flask, request, render_template, Response, send_file
from flask_apscheduler import APScheduler
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from helper import get_snnTorch_preds, create_figure
from scraper import Scraper
import numpy as np

app = Flask(__name__, template_folder='templates')
scheduler = APScheduler()
scrape_every = 60

ticks = 'msft aapl goog'
cols = ['Open','High','Low','Close','Adj Close','Volume']
sc = Scraper()
sc.set_tickers(ticks)
sc.set_cols(cols)

interval='1d'
start_date='2019-12-11'
end_date='2020-09-24'
pred_end_date='2020-12-10'



#plots ticker history
@app.route('/image.png', methods=("POST", "GET"))
@scheduler.task(id = 'image', trigger="interval", seconds=scrape_every)
def image():
    df = sc.get_ticker_hist(sc._rand, start=start_date, end=pred_end_date, interval=interval)
    with app.app_context():
        pdf, loss = get_snnTorch_preds(df)
        fig = create_figure(pdf, col='forecast_vol', title=sc._rand)
        fig.savefig('ticker_forecasts.png')
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')


#plots predicted ticker prices
@app.route('/image2.png', methods=("POST", "GET"))
@scheduler.task(id = 'image2', trigger="interval", seconds=scrape_every)
def image2():
    df = sc.get_ticker_hist(sc._rand, start=start_date, end=pred_end_date, interval=interval)
    with app.app_context():
        ndf = df.loc[:pred_end_date]
        fig = create_figure(ndf[['Volume']], col='ground_vol', title=sc._rand)
        fig.savefig('ticker_history.png')
        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        return Response(output.getvalue(), mimetype='image/png')



#posts ticker history as table in flask app
@app.route('/', methods=("POST", "GET"))
@scheduler.task(id = 'Scheduled Task', trigger="interval", seconds=scrape_every)
def scheduleTask():
    sc.rand_tick()
    print(sc._rand)
    df = sc.get_ticker_hist(sc._rand, start=start_date, end=pred_end_date, interval=interval)
    with app.app_context():
        df['Date'] = df.index
        df.reset_index(drop=True, inplace=True)
        df = df[['Date',*cols]].sort_values(by='Date', ascending=False)
        return render_template('simple.html', column_names=df.columns.values, row_data=list(df.values.tolist()), zip=zip)



if __name__ == "__main__":
    scheduler.init_app(app)
    scheduler.start()
    app.run()