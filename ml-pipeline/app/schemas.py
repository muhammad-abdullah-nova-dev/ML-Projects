"""
Pydantic v2 schemas for Credit Risk Assessment API.
Includes input boundary validation, risk tiering, batch processing, and governance metadata.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class CreditApplicantFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    RevolvingUtilizationOfUnsecuredLines: float = Field(
        ..., ge=0.0, le=100.0, description="Revolving balance ratio", examples=[0.35]
    )
    age: int = Field(..., ge=18, le=120, description="Borrower age in years", examples=[45])
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(
        default=0, ge=0, le=98, alias="NumberOfTime30-59DaysPastDueNotWorse", examples=[0]
    )
    DebtRatio: float = Field(..., ge=0.0, le=100000.0, description="Debt to income ratio", examples=[0.25])
    MonthlyIncome: Optional[float] = Field(
        default=None, ge=0.0, description="Monthly income in USD (handles null values)", examples=[6500.0]
    )
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0, le=100, description="Active credit lines", examples=[8])
    NumberOfTimes90DaysLate: int = Field(default=0, ge=0, le=98, examples=[0])
    NumberRealEstateLoansOrLines: int = Field(default=0, ge=0, le=50, examples=[1])
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(
        default=0, ge=0, le=98, alias="NumberOfTime60-89DaysPastDueNotWorse", examples=[0]
    )
    NumberOfDependents: Optional[float] = Field(default=0.0, ge=0.0, le=20.0, examples=[1.0])


class CreditPredictionResponse(BaseModel):
    default_risk_score: float = Field(..., description="Estimated probability of default (0.0 to 1.0)")
    risk_tier: str = Field(..., description="Risk tier classification ('Low', 'Moderate', 'High')")
    approved: bool = Field(..., description="Automated underwriting recommendation")
    threshold_used: float = Field(..., description="Decision cutoff threshold applied")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")


class BatchCreditRequest(BaseModel):
    applicants: List[CreditApplicantFeatures] = Field(..., min_length=1, max_length=500)
    decision_threshold: float = Field(default=0.25, gt=0.0, lt=1.0, description="Risk threshold for approval")


class BatchCreditResponse(BaseModel):
    results: List[CreditPredictionResponse]
    total_processed: int
    approval_rate: float
    average_risk_score: float
    total_latency_ms: float


class ModelGovernanceResponse(BaseModel):
    model_name: str
    maintainer: str
    best_model_algorithm: str
    best_roc_auc: float
    feature_names: List[str]
    benchmark_summary: Dict[str, Any]
    status: str