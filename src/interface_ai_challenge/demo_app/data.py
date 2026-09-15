from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Account:
    type: str
    number: str
    balance: str
    currency: str


@dataclass(frozen=True)
class Member:
    id: str
    name: str
    ssn: str
    birth_date: str
    branch: str
    accounts: tuple[Account, ...]


SYNTHETIC_MEMBERS: dict[str, Member] = {
    member.id: member
    for member in (
        Member(
            id="M-10023",
            name="Alice Carter",
            ssn="***-**-4821",
            birth_date="1984-03-12",
            branch="Downtown",
            accounts=(
                Account("Checking", "••••7710", "$3,210.00", "USD"),
                Account("Savings", "••••4821", "$1,250.45", "USD"),
            ),
        ),
        Member(
            id="M-100231",
            name="Alan Carter",
            ssn="***-**-0937",
            birth_date="1979-11-02",
            branch="Riverside",
            accounts=(Account("Savings", "••••0937", "$88.10", "USD"),),
        ),
        Member(
            id="M-20417",
            name="Bruno Silva",
            ssn="***-**-5530",
            birth_date="1991-07-25",
            branch="Uptown",
            accounts=(
                Account("Checking", "••••2201", "$540.00", "USD"),
                Account("Savings", "••••5530", "$12,980.07", "USD"),
            ),
        ),
        Member(
            id="M-30555",
            name="Carla Nunes",
            ssn="***-**-1188",
            birth_date="1968-01-30",
            branch="Downtown",
            accounts=(Account("Checking", "••••6612", "$75.20", "USD"),),
        ),
    )
}
