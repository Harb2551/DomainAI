"""
Generates only normal (non-edge, non-inappropriate) synthetic domain name cases.
"""

class NormalCaseGenerator:
    def __init__(self, n=500):
        self.n = n
        self.business_types = list(set([
            "bakery", "tech startup", "law firm", "tattoo parlor", "organic coffee shop",
            "pet grooming service", "luxury car dealership", "children's bookstore", "yoga studio",
            "AI consulting firm", "vintage clothing store", "craft brewery", "wedding planner",
            "mobile app developer", "eco-friendly cleaning service", "financial advisor",
            "fitness center", "art gallery", "dental clinic", "food truck", "music school",
            "photography studio", "florist", "hardware store", "book publisher", "language school",
            "travel agency", "spa and wellness center", "sports equipment shop", "toy store",
            "interior design firm", "digital marketing agency", "pet adoption center", "ice cream shop",
            "farmers market", "bike repair shop", "craft supply store", "brewpub", "comic book shop",
            "thrift store", "organic farm", "juice bar", "record store", "gaming lounge",
            "consulting agency", "web design studio", "catering service", "event management company",
            "landscaping business", "car rental agency", "home cleaning service", "dog walking company",
            "personal trainer", "nutritionist", "accounting firm", "legal consultancy", "translation service",
            "market research firm", "public relations agency", "advertising company", "fashion boutique",
            "shoe store", "electronics shop", "mobile repair center", "furniture store", "gift shop",
            "stationery store", "optician", "watch repair shop", "jewelry store", "barbershop",
            "hair salon", "spa retreat", "wellness clinic", "orthodontist", "pediatrician office",
            "veterinary clinic", "pet hotel", "dog training school", "car wash", "auto repair shop",
            "motorcycle dealership", "bicycle shop", "sports bar", "nightclub", "music venue",
            "recording studio", "film production company", "art supply store", "gallery space",
            "museum shop", "science center", "planetarium", "children's museum", "escape room",
            "board game cafe", "arcade", "laser tag arena", "paintball park", "mini golf course"
        ]))
        self.adjectives = list(set([
            "organic", "modern", "family-owned", "downtown", "suburban", "online", "local",
            "international", "award-winning", "24/7", "boutique", "premium", "affordable",
            "sustainable", "creative", "luxury", "express", "digital", "classic", "innovative",
            "friendly", "trusted", "exclusive", "seasonal", "mobile", "urban", "rural",
            "artisanal", "eco-friendly", "community", "independent", "custom", "vintage",
            "fresh", "dynamic", "expert", "wellness", "holistic", "stylish", "fun", "unique",
            "professional", "efficient", "reliable", "fast", "convenient", "affectionate",
            "energetic", "inspiring", "motivational", "supportive", "knowledgeable", "experienced",
            "innovative", "cutting-edge", "trendy", "chic", "elegant", "sophisticated", "minimalist",
            "colorful", "playful", "adventurous", "outdoor", "indoor", "cozy", "spacious",
            "historic", "futuristic", "eco-conscious", "green", "resourceful", "caring", "compassionate",
            "dedicated", "skilled", "talented", "award-winning", "renowned", "notable", "famous",
            "legendary", "pioneering", "groundbreaking", "leading", "top-rated", "highly-rated"
        ]))
        self.locations = list(set([
            "in downtown area", "serving the city", "for kids", "for seniors", "in the suburbs",
            "with global reach", "in the tech district", "by the beach", "in the mountains",
            "near the airport", "in the shopping mall", "on main street", "in the business district",
            "by the river", "in the countryside", "in the historic quarter", "in the arts district",
            "close to university", "in the industrial park", "in the medical center", "in the sports complex",
            "in the entertainment zone", "in the science park", "in the food court", "in the city center",
            "in the old town", "in the new development", "in the marina", "in the park", "in the plaza",
            "in the market square", "in the shopping center", "in the residential area", "in the commercial hub"
        ]))
    def generate_domain_name(self, description):
        import random
        words = [w for w in description.lower().split() if w.isalnum()]
        n_words = random.choice([2, 3, 4])
        if len(words) < n_words:
            n_words = len(words)
        # Join words (with no truncation) to get a domain base of at least 15, ideally around 20 chars, but do not cut words
        domain_base = "".join(words[:n_words])
        i = n_words
        while len(domain_base) < 15 and i < len(words):
            domain_base += words[i]
            i += 1
        # If still less than 15, just use as many as possible
        # If longer than 20, keep as is (do not truncate mid-word)
        tld = random.choice([".com", ".net", ".org"])
        return domain_base + tld

    def generate(self):
        import random
        import pandas as pd
        data = []
        # Ensure a uniform length distribution from 10 to 120 characters
        min_len = 20
        max_len = 120
        import random
        for i in range(self.n):
            target_len = random.randint(min_len, max_len)
            desc = ""
            while len(desc) < target_len:
                adj = random.choice(self.adjectives)
                btype = random.choice(self.business_types)
                loc = random.choice(self.locations)
                chunk = f"{adj} {btype} {loc} "
                desc += chunk
            # No truncation of any word: finish the last chunk if it goes over target_len
            desc = desc.strip()
            domain = self.generate_domain_name(desc)
            data.append({
                "business_description": desc,
                "ideal_domain": domain,
                "label": "normal"
            })
        return pd.DataFrame(data)

    def save(self, path="synthetic_domain_dataset_normal.csv"):
        df = self.generate()
        df.to_csv(path, index=False)
        print(f"Normal dataset saved to {path}")
