"""
Data Sources for Synthetic Domain Generation
Contains all the lists and constants used for generating business descriptions and edge cases
"""

class GenerationData:
    """Class containing all data sources for synthetic generation"""
    
    # Curated Industries - exactly 50 unique business types
    INDUSTRIES = [
        "tech startup", "bakery", "law firm", "tattoo parlor", "coffee shop",
        "pet grooming service", "car dealership", "bookstore", "yoga studio", "consulting firm",
        "clothing store", "brewery", "wedding planner", "app developer", "cleaning service",
        "financial advisor", "fitness center", "art gallery", "dental clinic", "food truck",
        "music school", "photography studio", "florist", "hardware store", "book publisher",
        "language school", "travel agency", "spa center", "sports shop", "toy store",
        "design firm", "marketing agency", "pet adoption center", "ice cream shop", "farmers market",
        "repair shop", "craft store", "restaurant", "comic shop", "thrift store",
        "organic farm", "juice bar", "record store", "gaming lounge", "catering service",
        "landscaping business", "car rental", "personal trainer", "nutritionist", "accounting firm"
    ]
    
    # Curated Differentiators - exactly 20 unique modifiers
    DIFFERENTIATORS = [
        "sustainable", "luxury", "affordable", "innovative", "mobile",
        "premium", "eco-friendly", "custom", "express", "professional",
        "creative", "modern", "vintage", "high-tech", "exclusive",
        "family-owned", "award-winning", "cutting-edge", "holistic", "urban"
    ]
    
    # Curated Target Markets - exactly 20 unique customer segments
    TARGET_MARKETS = [
        "busy professionals", "health-conscious families", "tech enthusiasts", "small businesses", "millennials",
        "seniors", "urban dwellers", "pet owners", "fitness enthusiasts", "luxury clientele",
        "budget-conscious customers", "creative professionals", "entrepreneurs", "students", "parents",
        "remote workers", "freelancers", "startups", "healthcare providers", "educational institutions"
    ]
    
    # Inappropriate Keywords - expanded list
    INAPPROPRIATE_KEYWORDS = [
        "adult entertainment", "illegal substances", "fake documents", "pyramid scheme",
        "money laundering", "identity theft", "illegal weapons", "scam operation",
        "fraudulent investment", "blackmail services", "illegal gambling", "drug dealing",
        "nude photography", "escort services", "pornographic content", "sexual services",
        "human trafficking", "child exploitation", "prostitution services", "explicit content",
        "drug trafficking", "weapon smuggling", "terrorism financing", "counterfeit goods",
        "stolen merchandise", "credit card fraud", "tax evasion services", "bribery consulting",
        "ponzi scheme", "insider trading", "embezzlement services", "racketeering operation",
        "extortion business", "loan sharking", "unlicensed pharmacy", "bootleg alcohol",
        "contraband smuggling", "forged certificates", "illegal surveillance", "wiretapping services",
        "hacking services", "data theft", "ransomware operation", "phishing scam",
        "cryptocurrency fraud", "investment scam", "fake charity", "insurance fraud",
        "medicare fraud", "welfare fraud", "bankruptcy fraud", "mortgage fraud",
        "check kiting", "wire fraud", "mail fraud", "securities fraud"
    ]
    
    # Gibberish Samples - expanded list
    GIBBERISH_SAMPLES = [
        "qwertyuiop asdfghjkl", "zxcvbnm poiuytrewq", "blarghmuffin wibblewobble",
        "xyzqpw mnbvcxz", "qwerasdf zxcvtyui", "gibberishtext nonsensewords",
        "alakazam presto changeo", "hokuspokus abracadabra", "flibbertigibbet",
        "supercalifragilisticexpialidocious babble", "jabberwocky mimsy borogroves",
        "wibbledy wobbledy", "flipperty flopperty", "snickerdoodle frazzle",
        "bippity boppity", "razzmatazz flibberty", "gobbledygook balderdash",
        "fiddle faddle", "higgledy piggledy", "topsy turvy", "helter skelter",
        "mumbo jumbo", "hocus pocus", "hanky panky", "hokey pokey",
        "jibberty jabberty", "flim flam", "zig zag", "ping pong",
        "tic tac", "kit kat", "flip flop", "clip clop"
    ]
    
    # Very Long Words - expanded list
    LONG_WORDS = [
        "Lopadotemachoselachogaleokranioleipsanodrimypotrimmatosilphioparaomelitokatakechymenokichlepikossyphophattoperisteralektryonoptekephalliokigklopeleiolagoiosiraiobaphetraganopterygon",
        "hippopotomonstrosesquipedaliophobia",
        "floccinaucinihilipilification",
        "antidisestablishmentarianism",
        "Supercalifragilisticexpialidocious",
        "Pneumonoultramicroscopicsilicovolcanoconiosis",
        "pseudopseudohypoparathyroidism",
        "spectrophotofluorometrically",
        "thyroparathyroidectomized",
        "dichlorodifluoromethane",
        "electroencephalographically",
        "immunoelectrophoretically",
        "tetraiodophenolphthalein",
        "hepaticocholangiogastrostomy",
        "radioimmunoelectrophoresis",
        "pneumonoultramicroscopicsilicovolcanoconiosisosis",
        "cyclotrimethylenetrinitramine",
        "trinitrophenylmethylnitramine",
        "tetrachlorodibenzoparadioxin",
        "hexamethylenetetraminehexanitrate"
    ]
    
    # Special Characters - expanded list
    SPECIAL_CHARS = [
        "!@#", "$%^", "&*()", "[]{}", ":;'<>,.?/|\\", "~`", "+=", "-_", 
        "¡¢£¤¥", "§¨©ª", "«¬®¯", "°±²³", "´µ¶·", "¸¹º»", "¼½¾¿",
        "×÷", "αβγδ", "♠♣♥♦", "★☆", "→←↑↓", "∞∑∏", "≠≤≥"
    ]
    
    # Numbers List - expanded list
    NUMBERS = [
        2025, 7, 314, 123, 999, 42, 88, 56, 101, 77, 365, 24, 100, 500, 1000,
        2024, 2023, 2022, 2021, 2020, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        25, 30, 50, 75, 200, 250, 300, 400, 600, 700, 800, 900, 1001, 1234, 5678, 9999,
        111, 222, 333, 444, 555, 666, 777, 888, 1, 2, 3, 4, 5, 6, 21, 22, 23
    ]
    
    # Ambiguous Words - expanded list
    AMBIGUOUS_WORDS = [
        "???", "undefined", "random", "odd", "strange", "confusing",
        "mystery", "unknown", "enigmatic", "peculiar", "bizarre", "weird",
        "unusual", "curious", "puzzling", "baffling", "perplexing", "mystifying",
        "cryptic", "obscure", "vague", "ambiguous", "unclear", "indefinite",
        "indeterminate", "uncertain", "doubtful", "questionable", "suspicious", "fishy"
    ]
    
    # Single Words - expanded list
    SINGLE_WORDS = [
        "business", "company", "service", "shop", "store", "firm", "agency", "consulting", 
        "corporation", "enterprise", "organization", "group", "startup", "venture", "project",
        "initiative", "solution", "platform", "system", "network", "hub", "center", "institute",
        "foundation", "association", "society", "guild", "collective", "cooperative", "consortium",
        "syndicate", "alliance", "partnership", "collaboration", "union", "federation", "league",
        "studio", "workshop", "laboratory", "clinic", "practice", "office", "bureau", "department"
    ]