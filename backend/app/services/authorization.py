"""Transit vehicle exemption and classification logic for PMPML BRT corridors."""
from enum import Enum


class AuthStatus(str, Enum):
    PMPML_AUTHORIZED = "PMPML Bus - Authorized"
    PRIVATE = "Private - Unauthorized"
    NEEDS_REVIEW = "Manual Review Required"


def classify_authorization(
    category: str,
    class_conf: float,
    pmpml_confirmed: bool = False,
    min_conf: float = 0.30,
) -> AuthStatus:
    """Classify vehicle authorization from detection category, confidence, and fleet status."""
    normalized_category = category.strip().lower()

    if class_conf < min_conf:
        return AuthStatus.NEEDS_REVIEW

    if normalized_category == "bus":
        # Only confirmed PMPML fleet buses are whitelisted for dedicated BRT lane operation
        return AuthStatus.PMPML_AUTHORIZED if pmpml_confirmed else AuthStatus.NEEDS_REVIEW

    return AuthStatus.PRIVATE