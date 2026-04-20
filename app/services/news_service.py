import os
import json
import asyncio
import httpx
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

NEWSAPI_BASE       = "https://newsapi.org/v2/everything"
ALPHAVANTAGE_BASE  = "https://www.alphavantage.co/query"

# Cache so we only fetch the S&P 500 list once per server run
_SP500_MAP: Dict[str, str] = {}


# Static S&P 500 ticker to company name map (all 500+ companies)
# This avoids Wikipedia scraping which can be blocked (HTTP 403)
SP500_STATIC_MAP = {
    "A": "Agilent Technologies Inc.",
    "AAL": "American Airlines Group Inc.",
    "AAPL": "Apple Inc.",
    "ABBV": "AbbVie Inc.",
    "ABC": "AmerisourceBergen Corporation",
    "ABMD": "Abiomed Inc.",
    "ABT": "Abbott Laboratories",
    "ACN": "Accenture plc",
    "ADBE": "Adobe Inc.",
    "ADI": "Analog Devices Inc.",
    "ADM": "Archer Daniels Midland Company",
    "ADP": "Automatic Data Processing Inc.",
    "ADSK": "Autodesk Inc.",
    "AEE": "Ameren Corporation",
    "AEP": "American Electric Power Company Inc.",
    "AES": "AES Corporation",
    "AFL": "Aflac Incorporated",
    "AIG": "American International Group Inc.",
    "AIV": "Apartment Income REIT Corp.",
    "AIZ": "Assurant Inc.",
    "AJG": "Arthur J. Gallagher & Co.",
    "AKAM": "Akamai Technologies Inc.",
    "ALB": "Albemarle Corporation",
    "ALGN": "Align Technology Inc.",
    "ALK": "Alaska Air Group Inc.",
    "ALL": "Allstate Corporation",
    "ALLE": "Allegion plc",
    "AMAT": "Applied Materials Inc.",
    "AMCR": "Amcor plc",
    "AMD": "Advanced Micro Devices Inc.",
    "AME": "AMETEK Inc.",
    "AMG": "Affiliated Managers Group Inc.",
    "AMGN": "Amgen Inc.",
    "AMR": "American Airlines Group Inc.",
    "AMSF": "Ameris Bancorp",
    "AMT": "American Tower Corporation",
    "AMZN": "Amazon.com Inc.",
    "AN": "AutoNation Inc.",
    "ANSS": "ANSYS Inc.",
    "ANTM": "Elevance Health Inc.",
    "AON": "Aon plc",
    "AOS": "A.O. Smith Corporation",
    "APA": "APA Corporation",
    "APD": "Air Products and Chemicals Inc.",
    "APH": "Amphenol Corporation",
    "APTV": "Aptiv PLC",
    "ARE": "Alexandria Real Estate Equities Inc.",
    "ARNA": "Arena Pharmaceuticals Inc.",
    "ARW": "Arrow Electronics Inc.",
    "ASML": "ASML Holding N.V.",
    "ATAS": "Atlas Air Worldwide Holdings Inc.",
    "ATVI": "Activision Blizzard Inc.",
    "AVB": "AvalonBay Communities Inc.",
    "AVGO": "Broadcom Inc.",
    "AVY": "Avery Dennison Corporation",
    "AWK": "American Water Works Company Inc.",
    "AXP": "American Express Company",
    "AXS": "AXIS Capital Holdings Limited",
    "AYI": "Acuity Brands Inc.",
    "AZO": "AutoZone Inc.",
    "B": "Barnes Group Inc.",
    "BA": "Boeing Company",
    "BACC": "Bactes Imaging Solutions",
    "BAC": "Bank of America Corp.",
    "BAX": "Baxter International Inc.",
    "BBWI": "Bath & Body Works Inc.",
    "BBY": "Best Buy Co. Inc.",
    "BDC": "Bell Ringers Companies Inc.",
    "BDX": "Becton Dickinson and Company",
    "BEN": "Franklin Resources Inc.",
    "BIIB": "Biogen Inc.",
    "BILL": "Bill.com Holdings Inc.",
    "BIO": "Bio-Rad Laboratories Inc.",
    "BK": "Bank of New York Mellon Corporation",
    "BKNG": "Booking Holdings Inc.",
    "BKR": "Baker Hughes Company",
    "BLK": "BlackRock Inc.",
    "BLL": "Ball Corporation",
    "BMY": "Bristol-Myers Squibb Company",
    "BND": "Vanguard Total Bond Market ETF",
    "BOH": "Bank of Hawaii Corporation",
    "BOK": "BOK Financial Corporation",
    "BR": "Broadridge Financial Solutions Inc.",
    "BRK.A": "Berkshire Hathaway Inc.",
    "BRK.B": "Berkshire Hathaway Inc.",
    "BRO": "Brown & Brown Inc.",
    "BSX": "Boston Scientific Corporation",
    "BWA": "BorgWarner Inc.",
    "BXP": "Boston Properties Inc.",
    "C": "Citigroup Inc.",
    "CAG": "Conagra Brands Inc.",
    "CAH": "Cardinal Health Inc.",
    "CAM": "Cameron International Corporation",
    "CARR": "Carrier Global Corporation",
    "CAT": "Caterpillar Inc.",
    "CB": "Chubb Limited",
    "CBRE": "CBRE Group Inc.",
    "CCI": "Crown Castle International Corp.",
    "CCL": "Carnival Corporation",
    "CDNS": "Cadence Design Systems Inc.",
    "CDW": "CDW Corporation",
    "CE": "Celanese Corporation",
    "CEG": "Constellation Energy Corporation",
    "CF": "CF Industries Holdings Inc.",
    "CFG": "Citizens Financial Group Inc.",
    "CHD": "Church & Dwight Co. Inc.",
    "CHRW": "C. H. Robinson Worldwide Inc.",
    "CHTR": "Charter Communications Inc.",
    "CI": "Cigna Group",
    "CINF": "Cincinnati Financial Corporation",
    "CL": "Colgate-Palmolive Company",
    "CLX": "Clorox Company",
    "CMA": "Comerica Incorporated",
    "CMCSA": "Comcast Corporation",
    "CME": "CME Group Inc.",
    "CMG": "Chipotle Mexican Grill Inc.",
    "CMI": "Cummins Inc.",
    "CMS": "CMS Energy Corporation",
    "CNC": "Centene Corporation",
    "CNP": "CenterPoint Energy Inc.",
    "COF": "Capital One Financial Corporation",
    "COG": "Cabot Oil & Gas Corporation",
    "COO": "Cooper Companies Inc.",
    "COP": "ConocoPhillips",
    "COST": "Costco Wholesale Corp.",
    "COTY": "Coty Inc.",
    "CPB": "Campbell Soup Company",
    "CPNG": "Coupang Inc.",
    "CPRT": "Copart Inc.",
    "CRL": "Charles River Laboratories International Inc.",
    "CRM": "Salesforce Inc.",
    "CRS": "Carpenter Technology Corporation",
    "CSCO": "Cisco Systems Inc.",
    "CSX": "CSX Corporation",
    "CTAS": "Cintas Corporation",
    "CTLT": "Catalent Inc.",
    "CTVA": "Corteva Inc.",
    "CVS": "CVS Health Corporation",
    "CVX": "Chevron Corporation",
    "CWH": "Camping World Holdings Inc.",
    "CWK": "Cushman & Wakefield plc",
    "D": "Dominion Energy Inc.",
    "DAL": "Delta Air Lines Inc.",
    "DASH": "DoorDash Inc.",
    "DB": "Deutsche Bank AG",
    "DD": "DuPont de Nemours Inc.",
    "DE": "Deere & Company",
    "DELL": "Dell Technologies Inc.",
    "DEO": "Diageo plc",
    "DFS": "Discover Financial Services",
    "DG": "Dollar General Corporation",
    "DGX": "Quest Diagnostics Incorporated",
    "DHI": "D.R. Horton Inc.",
    "DHR": "Danaher Corporation",
    "DIS": "Walt Disney Company",
    "DISH": "DISH Network Corporation",
    "DLR": "Digital Realty Trust Inc.",
    "DLTR": "Dollar Tree Inc.",
    "DOV": "Dover Corporation",
    "DOW": "Dow Inc.",
    "DPZ": "Domino's Pizza Inc.",
    "DRI": "Darden Restaurants Inc.",
    "DTE": "DTE Energy Company",
    "DTM": "Delta Air Lines",
    "DUK": "Duke Energy Corporation",
    "DVA": "DaVita Inc.",
    "DVN": "Devon Energy Corporation",
    "DXC": "DXC Technology Company",
    "DXCM": "DexCom Inc.",
    "EA": "Electronic Arts Inc.",
    "EBAY": "eBay Inc.",
    "ECL": "Ecolab Inc.",
    "ED": "Consolidated Edison Inc.",
    "EFX": "Equifax Inc.",
    "EIX": "Edison International",
    "EL": "Estee Lauder Companies Inc.",
    "ELAN": "Elanco Animal Health Incorporated",
    "ELV": "Elevance Health Inc.",
    "EMN": "Eastman Chemical Company",
    "EMR": "Emerson Electric Co.",
    "ENPH": "Enphase Energy Inc.",
    "EOG": "EOG Resources Inc.",
    "EPAM": "EPAM Systems Inc.",
    "EPC": "Edgewell Personal Care Company",
    "EQIX": "Equinix Inc.",
    "EQR": "Equity Residential",
    "EQT": "EQT Corporation",
    "ES": "Eversource Energy",
    "ESS": "Essex Property Trust Inc.",
    "ETN": "Eaton Corporation plc",
    "ETR": "Entergy Corporation",
    "ETSY": "Etsy Inc.",
    "EVRG": "Evergy Inc.",
    "EW": "Edwards Lifesciences Corporation",
    "EXC": "Exelon Corporation",
    "EXPD": "Expeditors International of Washington Inc.",
    "EXPE": "Expedia Group Inc.",
    "EXR": "Extra Space Storage Inc.",
    "F": "Ford Motor Company",
    "FANG": "Diamondback Energy Inc.",
    "FAST": "Fastenal Company",
    "FB": "Meta Platforms Inc.",
    "FBHS": "Fortune Brands Innovations Inc.",
    "FCX": "Freeport-McMoRan Inc.",
    "FDS": "FactSet Research Systems Inc.",
    "FDX": "FedEx Corporation",
    "FE": "FirstEnergy Corp.",
    "FFIV": "F5 Inc.",
    "FIS": "Fidelity National Information Services Inc.",
    "FISV": "Fiserv Inc.",
    "FITB": "Fifth Third Bancorp",
    "FLT": "FleetCor Technologies Inc.",
    "FLU": "Fluigent",
    "FMC": "FMC Corporation",
    "FOX": "Fox Corporation",
    "FOXA": "Fox Corporation",
    "FRC": "First Republic Bank",
    "FRT": "Federal Realty Investment Trust",
    "FSLR": "First Solar Inc.",
    "FTNT": "Fortinet Inc.",
    "FTV": "Fortive Corporation",
    "GILD": "Gilead Sciences Inc.",
    "GIS": "General Mills Inc.",
    "GL": "Globe Life Inc.",
    "GLW": "Corning Incorporated",
    "GM": "General Motors Company",
    "GPC": "Genuine Parts Company",
    "GPN": "Global Payments Inc.",
    "GPS": "Gap Inc.",
    "GRMN": "Garmin Ltd.",
    "GS": "Goldman Sachs Group Inc.",
    "GWW": "W.W. Grainger Inc.",
    "HAL": "Halliburton Company",
    "HAS": "Hasbro Inc.",
    "HBAN": "Huntington Bancshares Incorporated",
    "HCA": "HCA Healthcare Inc.",
    "HCP": "Healthpeak Properties Inc.",
    "HD": "Home Depot Inc.",
    "HES": "Hess Corporation",
    "HIG": "Hartford Financial Services Group Inc.",
    "HII": "Huntington Ingalls Industries Inc.",
    "HLT": "Hilton Worldwide Holdings Inc.",
    "HLY": "Hallador Energy Company",
    "HMN": "Horace Mann Educators Corporation",
    "HOMB": "Home BancShares Inc.",
    "HON": "Honeywell International Inc.",
    "HPE": "Hewlett Packard Enterprise Company",
    "HPQ": "HP Inc.",
    "HRL": "Hormel Foods Corporation",
    "HSIC": "Henry Schein Inc.",
    "HST": "Host Hotels & Resorts Inc.",
    "HSY": "Hershey Company",
    "HUM": "Humana Inc.",
    "HWM": "Howmet Aerospace Inc.",
    "IBM": "IBM Corporation",
    "ICE": "Intercontinental Exchange Inc.",
    "IDXX": "IDEXX Laboratories Inc.",
    "IEX": "IDEX Corporation",
    "IFF": "International Flavors & Fragrances Inc.",
    "ILMN": "Illumina Inc.",
    "INCY": "Incyte Corporation",
    "INFO": "IHS Markit Ltd.",
    "INTC": "Intel Corporation",
    "INTU": "Intuit Inc.",
    "INVH": "Invitation Homes Inc.",
    "IO": "IonQ Inc.",
    "IP": "International Paper Company",
    "IPG": "Interpublic Group of Companies Inc.",
    "IQV": "IQVIA Holdings Inc.",
    "IR": "Ingersoll Rand Inc.",
    "IRM": "Iron Mountain Incorporated",
    "ISRG": "Intuitive Surgical Inc.",
    "IT": "Gartner Inc.",
    "ITW": "Illinois Tool Works Inc.",
    "IVZ": "Invesco Ltd.",
    "J": "Jacobs Solutions Inc.",
    "JBHT": "J.B. Hunt Transport Services Inc.",
    "JCI": "Johnson Controls International plc",
    "JKHY": "Jack Henry & Associates Inc.",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase & Co.",
    "K": "Kellogg Company",
    "KDP": "Keurig Dr Pepper Inc.",
    "KEY": "KeyCorp",
    "KEYS": "Keysight Technologies Inc.",
    "KHC": "Kraft Heinz Company",
    "KIM": "Kimco Realty Corporation",
    "KLAC": "KLA Corporation",
    "KMB": "Kimberly-Clark Corporation",
    "KMI": "Kinder Morgan Inc.",
    "KMX": "CarMax Inc.",
    "KO": "Coca-Cola Company",
    "KRC": "Kilroy Realty Corporation",
    "KSM": "KSM Investment Advisors",
    "L": "Lazard Inc.",
    "LDOS": "Leidos Holdings Inc.",
    "LEG": "Leggett & Platt Incorporated",
    "LEN": "Lennar Corporation",
    "LH": "Laboratory Corporation of America Holdings",
    "LHX": "L3Harris Technologies Inc.",
    "LIN": "Linde plc",
    "LKQ": "LKQ Corporation",
    "LLY": "Eli Lilly and Company",
    "LMT": "Lockheed Martin Corporation",
    "LNC": "Lincoln National Corporation",
    "LNT": "Alliant Energy Corporation",
    "LOW": "Lowe's Companies Inc.",
    "LRCX": "Lam Research Corporation",
    "LSXMR": "BlackRock Institutional",
    "LUV": "Southwest Airlines Co.",
    "LVS": "Las Vegas Sands Corp.",
    "LW": "Lamb Weston Holdings Inc.",
    "LYB": "LyondellBasell Industries N.V.",
    "LYV": "Live Nation Entertainment Inc.",
    "M": "Macy's Inc.",
    "MA": "Mastercard Inc.",
    "MAA": "Mid-America Apartment Communities Inc.",
    "MAC": "Macerich Company",
    "MAR": "Marriott International Inc.",
    "MAS": "Masco Corporation",
    "MASI": "Masimo Corporation",
    "MCD": "McDonald's Corporation",
    "MCHP": "Microchip Technology Incorporated",
    "MCK": "McKesson Corporation",
    "MCO": "Moody's Corporation",
    "MDLZ": "Mondelez International Inc.",
    "MDT": "Medtronic plc",
    "MET": "MetLife Inc.",
    "META": "Meta Platforms Inc.",
    "MGM": "MGM Resorts International",
    "MHK": "Mohawk Industries Inc.",
    "MKC": "McCormick & Company Incorporated",
    "MKL": "Markel Group Inc.",
    "MLM": "Martin Marietta Materials Inc.",
    "MMC": "Marsh & McLennan Companies Inc.",
    "MMM": "3M Company",
    "MO": "Altria Group Inc.",
    "MOH": "Molina Healthcare Inc.",
    "MOS": "Mosaic Company",
    "MPC": "Marathon Petroleum Corporation",
    "MPWR": "Monolithic Power Systems Inc.",
    "MRK": "Merck & Co. Inc.",
    "MRNA": "Moderna Inc.",
    "MRO": "Marathon Oil Corporation",
    "MS": "Morgan Stanley",
    "MSCI": "MSCI Inc.",
    "MSFT": "Microsoft Corporation",
    "MSI": "Motorola Solutions Inc.",
    "MTB": "M&T Bank Corporation",
    "MTCH": "Match Group Inc.",
    "MTD": "Mettler-Toledo International Inc.",
    "MU": "Micron Technology Inc.",
    "MUR": "Murphy Oil Corporation",
    "MWA": "Mueller Water Products Inc.",
    "N": "NetApp Inc.",
    "NDAQ": "Nasdaq Inc.",
    "NDSN": "Nordson Corporation",
    "NEE": "NextEra Energy Inc.",
    "NEM": "Newmont Corporation",
    "NFLX": "Netflix Inc.",
    "NI": "NiSource Inc.",
    "NKE": "Nike Inc.",
    "NOC": "Northrop Grumman Corporation",
    "NOW": "ServiceNow Inc.",
    "NRG": "NRG Energy Inc.",
    "NSC": "Norfolk Southern Corporation",
    "NTAP": "NetApp Inc.",
    "NTRS": "Northern Trust Corporation",
    "NUE": "Nucor Corporation",
    "NVDA": "NVIDIA Corporation",
    "NVR": "NVR Inc.",
    "NWL": "Newell Brands Inc.",
    "O": "Realty Income Corporation",
    "ODFL": "Old Dominion Freight Line Inc.",
    "OHI": "Omega Healthcare Investors Inc.",
    "OKE": "ONEOK Inc.",
    "OMC": "Omnicom Group Inc.",
    "ON": "ON Semiconductor Corporation",
    "ORCL": "Oracle Corporation",
    "ORLY": "O'Reilly Automotive Inc.",
    "OSK": "Oshkosh Corporation",
    "OXY": "Occidental Petroleum Corporation",
    "PANW": "Palo Alto Networks Inc.",
    "PARA": "Paramount Global",
    "PAYC": "Paycom Software Inc.",
    "PAYX": "Paychex Inc.",
    "PCAR": "PACCAR Inc.",
    "PCG": "PG&E Corporation",
    "PEAK": "Healthpeak Properties",
    "PEG": "Public Service Enterprise Group Incorporated",
    "PEP": "PepsiCo Inc.",
    "PFE": "Pfizer Inc.",
    "PFG": "Principal Financial Group Inc.",
    "PG": "Procter & Gamble Co.",
    "PGR": "Progressive Corporation",
    "PH": "Parker-Hannifin Corporation",
    "PHM": "PulteGroup Inc.",
    "PKG": "Packaging Corporation of America",
    "PKI": "PerkinElmer Inc.",
    "PLD": "Prologis Inc.",
    "PLTR": "Palantir Technologies Inc.",
    "PM": "Philip Morris International Inc.",
    "PNC": "PNC Financial Services Group Inc.",
    "PNR": "Pentair plc",
    "PNW": "Pinnacle West Capital Corporation",
    "POOL": "Pool Corporation",
    "PPG": "PPG Industries Inc.",
    "PPL": "PPL Corporation",
    "PRU": "Prudential Financial Inc.",
    "PSA": "Public Storage",
    "PSX": "Phillips 66",
    "PTC": "PTC Inc.",
    "PWR": "Quanta Services Inc.",
    "PYPL": "PayPal Holdings Inc.",
    "QCOM": "QUALCOMM Inc.",
    "QRVO": "Qorvo Inc.",
    "R": "Ryder System Inc.",
    "RAI": "Renaissance Technologies",
    "RCL": "Royal Caribbean Cruises Ltd.",
    "REG": "Regency Centers Corporation",
    "REGN": "Regeneron Pharmaceuticals Inc.",
    "RHI": "Robert Half International Inc.",
    "RJF": "Raymond James Financial Inc.",
    "RL": "Ralph Lauren Corporation",
    "RMD": "ResMed Inc.",
    "ROK": "Rockwell Automation Inc.",
    "ROL": "Rollins Inc.",
    "ROP": "Roper Technologies Inc.",
    "ROST": "Ross Stores Inc.",
    "RS": "Reliance Steel & Aluminum Co.",
    "RTX": "RTX Corporation",
    "RIVN": "Rivian Automotive Inc.",
    "SBAC": "SBA Communications Corporation",
    "SBUX": "Starbucks Corporation",
    "SCHW": "Charles Schwab Corporation",
    "SEDG": "SolarEdge Technologies Inc.",
    "SEE": "Sealed Air Corporation",
    "SHW": "Sherwin-Williams Company",
    "SIRI": "Sirius XM Holdings Inc.",
    "SJM": "J.M. Smucker Company",
    "SLB": "Schlumberger Limited",
    "SNA": "Snap-on Incorporated",
    "SNAP": "Snap Inc.",
    "SNPS": "Synopsys Inc.",
    "SO": "Southern Company",
    "SPG": "Simon Property Group Inc.",
    "SPGI": "S&P Global Inc.",
    "SRE": "Sempra Energy",
    "STZ": "Constellation Brands Inc.",
    "SWK": "Stanley Black & Decker Inc.",
    "SWKS": "Skyworks Solutions Inc.",
    "SYF": "Synchrony Financial",
    "SYK": "Stryker Corporation",
    "SYY": "Sysco Corporation",
    "T": "AT&T Inc.",
    "TAP": "Molson Coors Beverage Company",
    "TCF": "TCF Financial Corporation",
    "TGT": "Target Corporation",
    "TIF": "Tiffany & Co.",
    "TJX": "TJX Companies Inc.",
    "TMO": "Thermo Fisher Scientific Inc.",
    "TMUS": "T-Mobile US Inc.",
    "TROW": "T. Rowe Price Group Inc.",
    "TRV": "Travelers Companies Inc.",
    "TSCO": "Tractor Supply Company",
    "TSLA": "Tesla Inc.",
    "TSN": "Tyson Foods Inc.",
    "TT": "Trane Technologies plc",
    "TTWO": "Take-Two Interactive Software Inc.",
    "TWTR": "Twitter Inc.",
    "TXN": "Texas Instruments Inc.",
    "TXT": "Textron Inc.",
    "TYL": "Tyler Technologies Inc.",
    "UA": "Under Armour Inc.",
    "UAA": "Under Armour Inc.",
    "UAL": "United Airlines Holdings Inc.",
    "UDR": "UDR Inc.",
    "UHS": "Universal Health Services Inc.",
    "ULTA": "Ulta Beauty Inc.",
    "UNH": "UnitedHealth Group Inc.",
    "UNM": "Unum Group",
    "UNP": "Union Pacific Corporation",
    "UPS": "United Parcel Service Inc.",
    "URI": "United Rentals Inc.",
    "USB": "U.S. Bancorp",
    "V": "Visa Inc.",
    "VFC": "V.F. Corporation",
    "VICI": "VICI Properties Inc.",
    "VLO": "Valero Energy Corporation",
    "VMC": "Vulcan Materials Company",
    "VRSK": "Verisk Analytics Inc.",
    "VRSN": "VeriSign Inc.",
    "VRTX": "Vertex Pharmaceuticals Incorporated",
    "VTR": "Ventas Inc.",
    "VZ": "Verizon Communications Inc.",
    "W": "Weyerhaeuser Company",
    "WAB": "Westinghouse Air Brake Technologies Corporation",
    "WAT": "Waters Corporation",
    "WBA": "Walgreens Boots Alliance Inc.",
    "WBD": "Warner Bros. Discovery Inc.",
    "WCG": "WellCare Health Plans Inc.",
    "WDC": "Western Digital Corporation",
    "WEC": "WEC Energy Group Inc.",
    "WELL": "Welltower Inc.",
    "WFC": "Wells Fargo & Company",
    "WHR": "Whirlpool Corporation",
    "WLTW": "Willis Towers Watson Public Limited Company",
    "WM": "Waste Management Inc.",
    "WMB": "Williams Companies Inc.",
    "WMT": "Walmart Inc.",
    "WRB": "W.R. Berkley Corporation",
    "WRK": "WestRock Company",
    "WST": "West Pharmaceutical Services Inc.",
    "WTW": "Willis Towers Watson",
    "WY": "Weyerhaeuser Company",
    "WYNN": "Wynn Resorts Limited",
    "X": "United States Steel Corporation",
    "XEL": "Xcel Energy Inc.",
    "XLB": "Materials Select Sector SPDR Fund",
    "XLC": "Communication Services Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLK": "Technology Select Sector SPDR Fund",
    "XLP": "Consumer Staples Select Sector SPDR Fund",
    "XLRE": "Real Estate Select Sector SPDR Fund",
    "XLU": "Utilities Select Sector SPDR Fund",
    "XLV": "Health Care Select Sector SPDR Fund",
    "XLY": "Consumer Discretionary Select Sector SPDR Fund",
    "XOM": "Exxon Mobil Corporation",
    "XRAY": "DENTSPLY SIRONA Inc.",
    "XYL": "Xylem Inc.",
    "YUM": "Yum! Brands Inc.",
    "ZBH": "Zimmer Biomet Holdings Inc.",
    "ZBRA": "Zebra Technologies Corporation",
    "ZION": "Zions Bancorporation N.A.",
    "ZTS": "Zoetis Inc.",
}


def _load_sp500_map() -> dict[str, str]:
    path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sp500.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def get_sp500_map() -> Dict[str, str]:
    """
    Returns the S&P 500 ticker → company name map.
    Uses a static list to avoid Wikipedia scraping issues (403 Forbidden).
    """
    global _SP500_MAP
    if _SP500_MAP:
        return _SP500_MAP

    # Use static map instead of scraping Wikipedia
    _SP500_MAP = _load_sp500_map()
    logger.info(f"✅ Loaded {len(_SP500_MAP)} S&P 500 tickers from data/sp500.json")

    return _SP500_MAP


def get_company_name(ticker: str) -> str:
    """Return the company name for a ticker, or the ticker itself as fallback."""
    sp500 = get_sp500_map()
    return sp500.get(ticker.upper(), ticker.upper())


def get_all_sp500_tickers() -> List[str]:
    """Return the full list of S&P 500 ticker symbols."""
    return list(get_sp500_map().keys())


FINNHUB_BASE  = "https://finnhub.io/api/v1"
EDGAR_SEARCH  = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FILING  = "https://www.sec.gov"
EDGAR_HEADERS = {"User-Agent": "Signal-Trading-App research@signal.app"}  # required by SEC


class EdgarNewsService:
    """
    SEC EDGAR full-text search — free, no API key, no rate cap beyond 10 req/sec.
    Fetches 8-K (material events) and 10-Q (quarterly results) filings for a
    ticker over any historical date range going back to the 1990s.

    Each filing is treated as one "article":
      title       = form type + company name + filed date
      description = first 1500 chars of the filing's human-readable text
      published_at= filing date (aligns to the trading day the news hit)
      source      = "SEC EDGAR"

    These are then scored by FinBERT in the same pipeline as Finnhub articles.

    SEC fair-use policy: identify your app in the User-Agent header (done above).
    Rate limit: max 10 requests/second — we stay well under with 0.15s sleep.
    """

    # Forms that contain price-moving information
    FORMS = "8-K,10-Q,10-K"

    async def fetch_filings(
        self,
        ticker: str,
        from_date: str,   # "YYYY-MM-DD"
        to_date: str,     # "YYYY-MM-DD"
        forms: str = None,
        max_results: int = 200,
    ) -> List[Dict]:
        """
        Search EDGAR for filings mentioning `ticker` in the given date range.
        Returns normalised article-shaped dicts ready for FinBERT scoring.
        """
        forms = forms or self.FORMS
        params = {
            "q":         f'"{ticker}"',
            "dateRange": "custom",
            "startdt":   from_date,
            "enddt":     to_date,
            "forms":     forms,
            "_source":   "period_of_report,file_date,entity_name,form_type,file_num",
            "hits.hits.total.value": max_results,
            "hits.hits._source.period_of_report": 1,
        }

        async with httpx.AsyncClient(timeout=20, headers=EDGAR_HEADERS) as client:
            resp = await client.get(EDGAR_SEARCH, params=params)

        if resp.status_code == 429:
            await asyncio.sleep(5)
            raise ValueError("EDGAR rate limit — retry after pause")
        if resp.status_code != 200:
            raise ValueError(f"EDGAR search error {resp.status_code}: {resp.text[:200]}")

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return []

        logger.info(f"📄 EDGAR {ticker} ({from_date}→{to_date}): {len(hits)} filings found")

        articles = []
        for hit in hits[:max_results]:
            src = hit.get("_source", {})
            try:
                article = await self._parse_filing(src, ticker)
                if article:
                    articles.append(article)
                await asyncio.sleep(0.15)   # stay under 10 req/sec SEC limit
            except Exception as e:
                logger.debug(f"EDGAR parse failed for {ticker} filing: {e}")

        return articles

    async def _parse_filing(self, src: dict, ticker: str) -> Optional[Dict]:
        """
        Fetch the filing index page and extract human-readable text from
        the first .htm document. Truncate to 1500 chars for FinBERT.
        """
        file_date  = src.get("file_date", "")
        entity     = src.get("entity_name", ticker)
        form_type  = src.get("form_type", "8-K")
        accession  = src.get("accession_no", "").replace("-", "")
        cik        = str(src.get("file_num", "")).lstrip("0") or ""

        try:
            published_at = datetime.strptime(file_date[:10], "%Y-%m-%d")
        except ValueError:
            published_at = datetime.now(timezone.utc).replace(tzinfo=None)

        # Build filing index URL and fetch the document list
        filing_text = ""
        if accession and cik:
            index_url = (
                f"{EDGAR_FILING}/cgi-bin/browse-edgar"
                f"?action=getcompany&CIK={cik}"
                f"&type={form_type}&dateb=&owner=include&count=1"
            )
            try:
                async with httpx.AsyncClient(
                    timeout=15, headers=EDGAR_HEADERS, follow_redirects=True
                ) as client:
                    # Fetch the filing document directly via accession number
                    doc_url = (
                        f"{EDGAR_FILING}/Archives/edgar/data/{cik}/"
                        f"{accession}/{accession}-index.htm"
                    )
                    r = await client.get(doc_url)
                    if r.status_code == 200:
                        # Extract visible text — strip HTML tags simply
                        import re
                        text = re.sub(r"<[^>]+>", " ", r.text)
                        text = re.sub(r"\s+", " ", text).strip()
                        filing_text = text[:1500]
            except Exception as e:
                logger.debug(f"[{ticker}] EDGAR filing body fetch failed (title-only): {e}")

        title = f"{form_type} — {entity} ({file_date[:10]})"
        description = filing_text or f"{form_type} filing by {entity}"

        # Use accession-number-based URL as the unique identifier per filing
        if accession and cik:
            unique_url = f"{EDGAR_FILING}/Archives/edgar/data/{cik}/{accession}/{accession}-index.htm"
        else:
            # Fallback: embed ticker + date + form so different filings stay distinct
            unique_url = f"{EDGAR_FILING}/cgi-bin/browse-edgar?ticker={ticker}&form={form_type}&date={file_date[:10]}"

        return {
            "ticker":       ticker.upper(),
            "title":        title,
            "description":  description,
            "url":          unique_url,
            "source":       "SEC EDGAR",
            "published_at": published_at,
        }

    async def fetch_date_range(
        self,
        ticker: str,
        from_date: str,   # "YYYY-MM-DD"
        to_date: str,     # "YYYY-MM-DD"
    ) -> List[Dict]:
        """
        Fetch all 8-K and 10-Q filings for ticker between from_date and to_date.
        Splits into yearly chunks to keep result sets manageable.
        """
        all_articles: List[Dict] = []

        start = datetime.strptime(from_date, "%Y-%m-%d")
        end   = datetime.strptime(to_date,   "%Y-%m-%d")
        now   = datetime.now(timezone.utc).replace(tzinfo=None)
        end   = min(end, now)

        # Iterate year by year
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=365), end)
            from_str  = chunk_start.strftime("%Y-%m-%d")
            to_str    = chunk_end.strftime("%Y-%m-%d")
            try:
                filings = await self.fetch_filings(ticker, from_str, to_str)
                all_articles.extend(filings)
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"⚠️  EDGAR chunk failed ({ticker} {from_str}→{to_str}): {e}")
                await asyncio.sleep(2)
            chunk_start = chunk_end

        logger.info(
            f"📄 EDGAR {ticker} ({from_date}→{to_date}): "
            f"{len(all_articles)} total filings collected"
        )
        return all_articles


class FinnhubNewsService:
    """
    Finnhub company news — free tier: 60 req/min, no daily cap.
    Endpoint: GET /company-news?symbol=AAPL&from=2021-01-01&to=2021-12-31
    Returns articles with headline + summary (no sentiment — we run FinBERT).
    Goes back several years of history.
    """

    def __init__(self):
        self.api_key = settings.FINNHUB_API_KEY

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def fetch_articles(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
    ) -> List[Dict]:
        """
        Fetch articles for ticker between from_date and to_date (YYYY-MM-DD).
        Returns normalised article dicts (no sentiment scores — caller runs FinBERT).
        """
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY is not set in .env")

        params = {
            "symbol": ticker.upper(),
            "from": from_date,
            "to": to_date,
            "token": self.api_key,
        }

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f"{FINNHUB_BASE}/company-news", params=params)

        if resp.status_code == 429:
            raise ValueError("Finnhub rate limit hit — slow down requests")
        if resp.status_code != 200:
            raise ValueError(f"Finnhub error {resp.status_code}: {resp.text[:200]}")

        articles = resp.json()
        if not isinstance(articles, list):
            return []

        logger.info(f"📰 Finnhub {ticker} ({from_date}→{to_date}): {len(articles)} articles")
        return [self._parse(a, ticker) for a in articles]

    def _parse(self, raw: Dict, ticker: str) -> Dict:
        """Normalise a raw Finnhub article into our internal format."""
        ts = raw.get("datetime", 0)
        try:
            published_at = datetime.fromtimestamp(int(ts), tz=timezone.utc).replace(tzinfo=None)
        except Exception:
            published_at = datetime.now(timezone.utc).replace(tzinfo=None)

        return {
            "ticker": ticker.upper(),
            "title": raw.get("headline") or "",
            "description": raw.get("summary") or "",
            "url": raw.get("url") or "",
            "source": raw.get("source") or "Unknown",
            "published_at": published_at,
        }

    async def fetch_date_range(
        self,
        ticker: str,
        years: int = 5,
        months_per_chunk: int = 1,
    ) -> tuple:
        """
        Fetch articles going backwards month by month from today.
        Stops early when 3 consecutive months return 0 articles —
        this is the free tier historical boundary (~1 year back).

        Returns:
            (articles: List[Dict], cutoff_date: str)
            cutoff_date is the earliest date successfully covered by Finnhub
            so the caller knows where to start the EDGAR fallback from.
        """
        all_articles: List[Dict] = []
        now = datetime.now(timezone.utc)
        cutoff_date = now.strftime("%Y-%m-%d")   # will be updated as we go back

        total_months = years * 12
        consecutive_empty = 0
        EMPTY_STOP_THRESHOLD = 3

        for month_offset in range(total_months):
            chunk_end   = now - timedelta(days=30 * month_offset)
            chunk_start = chunk_end - timedelta(days=30 * months_per_chunk)
            from_str    = chunk_start.strftime("%Y-%m-%d")
            to_str      = chunk_end.strftime("%Y-%m-%d")
            try:
                articles = await self.fetch_articles(ticker, from_str, to_str)
                if articles:
                    all_articles.extend(articles)
                    cutoff_date = from_str      # furthest back we got real data
                    consecutive_empty = 0
                else:
                    consecutive_empty += 1
                    if consecutive_empty >= EMPTY_STOP_THRESHOLD:
                        logger.info(
                            f"📰 Finnhub {ticker}: free tier limit reached at ~{from_str} "
                            f"({len(all_articles)} articles collected). "
                            f"EDGAR will cover the remaining history."
                        )
                        break
                await asyncio.sleep(1.1)   # respect 60 req/min
            except Exception as e:
                logger.warning(f"⚠️  Finnhub chunk failed ({ticker} {from_str}→{to_str}): {e}")
                await asyncio.sleep(2)

        return all_articles, cutoff_date


class NewsService:
    def __init__(self):
        self.api_key = settings.NEWSAPI_KEY

    async def fetch_articles(self, ticker: str, limit: int = 10) -> List[Dict]:
        """
        Fetch recent financial news articles for a ticker from NewsAPI.
        Uses the company name from the S&P 500 list for a better search query.
        """
        if not self.api_key:
            raise ValueError("NEWSAPI_KEY is not set in .env")

        company_name = get_company_name(ticker.upper())

        # Use company name if it differs from ticker (more readable queries)
        query = company_name if company_name != ticker.upper() else ticker.upper()

        # Free tier supports last 30 days; use 7 for freshness
        from_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime(
            "%Y-%m-%d"
        )

        params = {
            "q": f"{query} stock",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(limit, 100),
            "from": from_date,
            "apiKey": self.api_key,
        }

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(NEWSAPI_BASE, params=params)

        if response.status_code != 200:
            logger.error(f"NewsAPI error {response.status_code}: {response.text}")
            raise ValueError(f"NewsAPI request failed: {response.status_code}")

        data = response.json()
        if data.get("status") != "ok":
            raise ValueError(f"NewsAPI error: {data.get('message', 'unknown')}")

        articles = data.get("articles", [])
        logger.info(f"📰 {ticker} ({company_name}): fetched {len(articles)} articles")
        return articles

    def parse_article(self, raw: Dict, ticker: str) -> Dict:
        """Normalise a raw NewsAPI article into a clean dict."""
        published_raw = raw.get("publishedAt", "")
        try:
            published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            published_at = published_at.replace(tzinfo=None)
        except Exception:
            published_at = datetime.now(timezone.utc)

        return {
            "ticker": ticker.upper(),
            "title": raw.get("title") or "",
            "description": raw.get("description") or "",
            "url": raw.get("url") or "",
            "source": raw.get("source", {}).get("name") or "Unknown",
            "published_at": published_at,
        }


# ── Alpha Vantage News + Sentiment ───────────────────────────────────────────

class AlphaVantageNewsService:
    """
    Uses the Alpha Vantage NEWS_SENTIMENT endpoint.

    Free tier: 25 calls/day.
    Each call returns up to 200 articles with pre-computed sentiment scores
    (no FinBERT needed for these).

    Historical range:  supply time_from / time_to (format: YYYYMMDDTHHMM).
    Recent articles:   omit time_from / time_to.
    """

    # AV sentiment label → our canonical label
    _LABEL_MAP = {
        "Bullish":          "positive",
        "Somewhat-Bullish": "positive",
        "Neutral":          "neutral",
        "Somewhat-Bearish": "negative",
        "Bearish":          "negative",
    }

    def __init__(self):
        self.api_key = settings.ALPHAVANTAGE_KEY

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def fetch_articles(
        self,
        ticker: str,
        time_from: Optional[str] = None,
        time_to:   Optional[str] = None,
        limit:     int = 50,
    ) -> List[Dict]:
        """
        Fetch articles from Alpha Vantage for *ticker*.

        Args:
            ticker:    e.g. "AAPL"
            time_from: "YYYYMMDDTHHMM"  (e.g. "20220101T0000")
            time_to:   "YYYYMMDDTHHMM"
            limit:     max articles to return (AV max = 200 per call)

        Returns a list of normalised article dicts ready for the DB.
        """
        if not self.api_key:
            raise ValueError("ALPHAVANTAGE_KEY is not set in .env")

        params: Dict = {
            "function": "NEWS_SENTIMENT",
            "tickers":  ticker.upper(),
            "limit":    min(limit, 200),
            "apikey":   self.api_key,
            "sort":     "LATEST",
        }
        if time_from:
            params["time_from"] = time_from
        if time_to:
            params["time_to"] = time_to

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(ALPHAVANTAGE_BASE, params=params)

        if resp.status_code != 200:
            raise ValueError(f"Alpha Vantage API error {resp.status_code}: {resp.text[:200]}")

        data = resp.json()

        # AV returns {"Information": "..."} when rate-limited or key invalid
        if "Information" in data:
            raise ValueError(f"Alpha Vantage rate-limit / key error: {data['Information']}")
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")

        feed = data.get("feed", [])
        logger.info(f"📰 AV {ticker}: {len(feed)} articles fetched")
        return [self._parse(article, ticker) for article in feed]

    def _parse(self, raw: Dict, ticker: str) -> Dict:
        """
        Convert a raw Alpha Vantage article to our internal format,
        including pre-computed sentiment scores.
        """
        # Parse timestamp: "20231015T143000"
        ts_raw = raw.get("time_published", "")
        try:
            published_at = datetime.strptime(ts_raw, "%Y%m%dT%H%M%S")
        except ValueError:
            try:
                published_at = datetime.strptime(ts_raw[:15], "%Y%m%dT%H%M%S")
            except Exception:
                published_at = datetime.now(timezone.utc).replace(tzinfo=None)

        # Find ticker-specific sentiment (prefer over overall)
        ticker_upper = ticker.upper()
        ts_list = raw.get("ticker_sentiment", [])
        ts_entry = next(
            (t for t in ts_list if t.get("ticker", "").upper() == ticker_upper),
            None,
        )

        if ts_entry:
            score = float(ts_entry.get("ticker_sentiment_score", 0.0))
            label_raw = ts_entry.get("ticker_sentiment_label", "Neutral")
        else:
            score = float(raw.get("overall_sentiment_score", 0.0))
            label_raw = raw.get("overall_sentiment_label", "Neutral")

        label = self._LABEL_MAP.get(label_raw, "neutral")

        # Derive pos / neg / neutral scores from compound score (-1 → +1)
        pos = max(0.0, score)
        neg = max(0.0, -score)
        neu = round(1.0 - abs(score), 6)

        return {
            "ticker":          ticker_upper,
            "title":           raw.get("title") or "",
            "description":     raw.get("summary") or "",
            "url":             raw.get("url") or "",
            "source":          raw.get("source") or "Unknown",
            "published_at":    published_at,
            # Pre-computed sentiment — no FinBERT call needed
            "sentiment_label": label,
            "positive_score":  round(pos, 6),
            "negative_score":  round(neg, 6),
            "neutral_score":   round(neu, 6),
            "compound_score":  round(score, 6),
        }

    async def fetch_daily_sentiment_series(
        self,
        ticker: str,
        period_years: int = 2,
    ) -> Dict[str, float]:
        """
        Fetch enough historical articles to build a daily compound-score series.
        Returns {date_str: avg_compound_score} for each day that had articles.

        With AV free tier (25 calls/day) we fetch 2 calls per ticker:
          • last year
          • year before that
        This gives ~2 years of history.
        """
        results: Dict[str, List[float]] = {}
        now = datetime.now(timezone.utc)

        for yr_offset in range(period_years):
            t_end   = now - timedelta(days=365 * yr_offset)
            t_start = t_end - timedelta(days=365)
            tf = t_start.strftime("%Y%m%dT%H%M")
            tt = t_end.strftime("%Y%m%dT%H%M")
            try:
                articles = await self.fetch_articles(ticker, time_from=tf, time_to=tt, limit=200)
                for a in articles:
                    day = a["published_at"].strftime("%Y-%m-%d")
                    results.setdefault(day, []).append(a["compound_score"])
            except Exception as e:
                logger.warning(f"AV historical fetch failed ({ticker} yr-{yr_offset}): {e}")

        return {day: sum(scores) / len(scores) for day, scores in results.items()}
