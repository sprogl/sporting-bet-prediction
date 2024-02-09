import core.utils
import fastapi
import uvicorn

app = fastapi.FastAPI()


@app.get("/")
async def root():
    return "The app is up"


@app.get("/fetch")
async def fetch():
    try:
        core.utils.fetch()
    except:
        raise fastapi.HTTPException(status_code=500, detail="Something went wrong")


@app.post("/advise")
async def advise(p1: str, p2: str, field_type: str):
    try:
        elo_rates = core.utils.load()
        advise = core.utils.get_feats(
            P1_name=p1, P2_name=p2, field_type=field_type, elo_rates=elo_rates
        )
        return {
            "advise": advise["advise"].values[0],
            "predict_P1_wins": advise["predict_P1_wins"].values[0],
            "certainty": advise["certainty"].values[0],
        }
    except:
        raise fastapi.HTTPException(status_code=500, detail="Something went wrong")


@app.post("/fetch_advise")
async def fetch_advise(p1: str, p2: str, field_type: str):
    try:
        elo_rates = core.utils.fetch()
        advise = core.utils.get_feats(
            P1_name=p1, P2_name=p2, field_type=field_type, elo_rates=elo_rates
        )
        return {
            "advise": advise["advise"].values[0],
            "predict_P1_wins": advise["predict_P1_wins"].values[0],
            "certainty": advise["certainty"].values[0],
        }
    except:
        raise fastapi.HTTPException(status_code=500, detail="Something went wrong")


if __name__ == "__main__":
    config = uvicorn.Config("main:app", port=80, log_level="info")
    server = uvicorn.Server(config)
    server.run()
