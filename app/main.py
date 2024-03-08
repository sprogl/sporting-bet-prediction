import app.core.utils
import fastapi
import uvicorn

api = fastapi.FastAPI()


@api.get("/")
async def root():
    return "The application is up"


@api.get("/fetch")
async def fetch():
    try:
        app.core.utils.fetch()
    except:
        raise fastapi.HTTPException(status_code=500, detail="Something went wrong")


@api.post("/advise")
async def advise(p1: str, p2: str, field_type: str):
    try:
        elo_rates = app.core.utils.load()
        feats = app.core.utils.get_feats(
            P1_name=p1, P2_name=p2, field_type=field_type, elo_rates=elo_rates
        )
        df_advise = app.core.utils.get_advise(input=feats).fillna(value="None")
        return {
            "advise": df_advise["advise"].values[0],
            "predicted_winner": df_advise["predict_P1_wins"]
            .replace({True: p1, False: p2})
            .values[0],
            "certainty": round(df_advise["certainty"].values[0] * 100, 1),
        }
    except Exception as err:
        raise fastapi.HTTPException(
            status_code=500, detail=f"Something went wrong: {err}"
        )


@api.post("/fetch_advise")
async def fetch_advise(p1: str, p2: str, field_type: str):
    try:
        elo_rates = app.core.utils.fetch()
        feats = app.core.utils.get_feats(
            P1_name=p1, P2_name=p2, field_type=field_type, elo_rates=elo_rates
        )
        df_advise = app.core.utils.get_advise(input=feats).fillna(value="None")
        return {
            "advise": df_advise["advise"].values[0],
            "predicted_winner": df_advise["predict_P1_wins"]
            .replace({True: p1, False: p2})
            .values[0],
            "certainty": round(df_advise["certainty"].values[0] * 100, 1),
        }
    except Exception as err:
        raise fastapi.HTTPException(
            status_code=500, detail=f"Something went wrong: {err}"
        )


if __name__ == "__main__":
    config = uvicorn.Config("main:api", port=80, log_level="info")
    server = uvicorn.Server(config)
    server.run()
