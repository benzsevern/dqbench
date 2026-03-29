"""Shared fake data pools for dataset generators."""

FIRST_NAMES = [
    "James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda",
    "William", "Barbara", "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Lisa", "Daniel", "Nancy",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
    "Steven", "Dorothy", "Paul", "Kimberly", "Andrew", "Emily", "Joshua", "Donna",
    "Kenneth", "Michelle", "Kevin", "Carol", "Brian", "Amanda", "George", "Melissa",
    "Timothy", "Deborah", "Ronald", "Stephanie", "Edward", "Rebecca", "Jason", "Sharon",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
    "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Phillips", "Evans", "Turner", "Parker", "Collins", "Edwards",
]

DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "company.com", "work.org"]

CITIES = [
    ("New York", "NY", "10001"),
    ("Los Angeles", "CA", "90001"),
    ("Chicago", "IL", "60601"),
    ("Houston", "TX", "77001"),
    ("Phoenix", "AZ", "85001"),
    ("Philadelphia", "PA", "19101"),
    ("San Antonio", "TX", "78201"),
    ("San Diego", "CA", "92101"),
    ("Dallas", "TX", "75201"),
    ("San Jose", "CA", "95101"),
    ("Austin", "TX", "73301"),
    ("Jacksonville", "FL", "32099"),
    ("Fort Worth", "TX", "76101"),
    ("Columbus", "OH", "43085"),
    ("Charlotte", "NC", "28201"),
    ("Indianapolis", "IN", "46201"),
    ("San Francisco", "CA", "94101"),
    ("Seattle", "WA", "98101"),
    ("Denver", "CO", "80201"),
    ("Nashville", "TN", "37201"),
    ("Oklahoma City", "OK", "73101"),
    ("El Paso", "TX", "79901"),
    ("Boston", "MA", "02101"),
    ("Portland", "OR", "97201"),
    ("Las Vegas", "NV", "89101"),
]

COUNTRIES = ["US", "CA", "GB", "DE", "FR", "AU", "JP", "BR", "IN", "MX", "IT", "ES", "NL", "SE", "NO"]

STATUSES = ["active", "inactive", "pending"]

ACCOUNT_TYPES = ["basic", "premium", "enterprise"]

REFERRAL_SOURCES = [
    "Google", "google", "GOOGLE",
    "Facebook", "facebook", "FACEBOOK",
    "Twitter", "Direct", "Email",
    "Referral", "LinkedIn", "Instagram",
]

FREE_TEXT_NOTES = [
    "Prefers email contact",
    "Do not call before 9am",
    "VIP customer",
    "Has outstanding balance",
    "Requested paper invoices",
    "International shipping only",
    "Discount applied",
    "Follow up needed",
    "Account under review",
    "Longtime customer since 2015",
]

# ---- ER data pools ----

COMPANIES = [
    "Acme Corp", "Globex", "Initech", "Umbrella Inc", "Stark Industries",
    "Wayne Enterprises", "Cyberdyne", "Soylent Corp", "Oscorp", "LexCorp",
    "Wonka Industries", "Aperture Science", "Tyrell Corp", "Weyland-Yutani",
    "Massive Dynamic", "Hooli", "Pied Piper", "Dunder Mifflin", "Sterling Cooper",
    "Prestige Worldwide",
]

STREET_NAMES = [
    "Main St", "Oak Ave", "Elm St", "Park Blvd", "Maple Dr",
    "Cedar Ln", "Pine Rd", "Washington St", "Lake Ave", "Hill St",
    "River Rd", "Spring St", "Forest Dr", "Sunset Blvd", "Highland Ave",
    "Broadway", "Market St", "Church St", "Mill Rd", "Center St",
]

PHONE_AREA_CODES = [
    "212", "310", "312", "415", "512", "617", "702", "713", "718", "773",
    "818", "917", "202", "305", "404", "503", "602", "614", "704", "916",
]

# ---- ER Tier 2: Nickname mappings ----
NICKNAME_MAP: dict[str, list[str]] = {
    "Robert": ["Bob", "Rob", "Bobby"],
    "Elizabeth": ["Liz", "Beth", "Lizzy"],
    "William": ["Bill", "Will", "Billy"],
    "James": ["Jim", "Jimmy"],
    "Richard": ["Rick", "Dick", "Rich"],
    "Michael": ["Mike", "Mikey"],
    "Jennifer": ["Jen", "Jenny"],
    "Patricia": ["Pat", "Patty"],
    "Margaret": ["Meg", "Maggie", "Peggy"],
    "Joseph": ["Joe", "Joey"],
    "Thomas": ["Tom", "Tommy"],
    "Christopher": ["Chris"],
    "Daniel": ["Dan", "Danny"],
    "Matthew": ["Matt"],
    "Anthony": ["Tony"],
    "Steven": ["Steve"],
    "Kenneth": ["Ken", "Kenny"],
    "Timothy": ["Tim", "Timmy"],
    "Jessica": ["Jess", "Jessie"],
    "Barbara": ["Barb"],
}

# ---- ER Tier 3: Phonetic variants ----
PHONETIC_VARIANTS: dict[str, list[str]] = {
    "Smith": ["Smyth", "Smithe"],
    "Thompson": ["Thomson", "Tompson"],
    "Johnson": ["Johnsen", "Jonson"],
    "Williams": ["Willams", "Wiliams"],
    "Anderson": ["Andersen", "Andersson"],
    "Martinez": ["Martines", "Martinz"],
    "Wilson": ["Willson", "Wilsen"],
    "Taylor": ["Tailor", "Tayler"],
    "Moore": ["More", "Moor"],
    "Jackson": ["Jacksen", "Jaxon"],
}

ADDRESS_ABBREVIATIONS: dict[str, str] = {
    "St": "Street",
    "Ave": "Avenue",
    "Blvd": "Boulevard",
    "Dr": "Drive",
    "Ln": "Lane",
    "Rd": "Road",
}
