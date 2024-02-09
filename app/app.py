import core.utils
import fastapi
import uvicorn

app = fastapi.FastAPI()


@app.get("/")
async def root():
    pass


@app.get("/fetch")
async def fetch():
    try:
        core.utils.fetch()
    except:
        pass


@app.post("/advise")
async def fetch():
    try:
        elo_rates = core.utils.load()
        advise = core.utils.get_feats(
            P1_name=p1, P2_name=p2, field_type=ft, elo_rates=elo_rates
        )
        return {
            "advise": advise["advise"].values[0],
            "predict_P1_wins": advise["predict_P1_wins"].values[0],
            "certainty": advise["certainty"].values[0],
        }
    except:
        pass


@app.post("/fetch_advise")
async def fetch():
    try:
        elo_rates = core.utils.fetch()
        advise = core.utils.get_feats(
            P1_name=p1, P2_name=p2, field_type=ft, elo_rates=elo_rates
        )
        return {
            "advise": advise["advise"].values[0],
            "predict_P1_wins": advise["predict_P1_wins"].values[0],
            "certainty": advise["certainty"].values[0],
        }
    except:
        pass


if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=80, log_level="info")
    server = uvicorn.Server(config)
    server.run()
