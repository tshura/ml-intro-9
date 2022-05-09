from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler



def create_pipeline(
    use_scaler: bool, s:str, m:str, max_iter: int, logreg_C: float, random_state: int, n:int, crit: str, m_depth: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        if s == 'ss':
            pipeline_steps.append(("scaler", StandardScaler()))
        elif s == 'mm':
            pipeline_steps.append(("scaler", MinMaxScaler()))
    if m == 'logreg':
        pipeline_steps.append(
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state, max_iter=max_iter, C=logreg_C
                ),
            )
        )
    elif m == 'rf':
        pipeline_steps.append(
            (
                "classifier",
                RandomForestClassifier(
                    random_state=random_state, max_depth=m_depth, n_estimators=n, criterion = crit
                ),
            )
        )
    
    return Pipeline(steps=pipeline_steps)
