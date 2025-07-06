"""
Generates only edge and inappropriate synthetic domain name cases for testing.
"""
import pandas as pd

class EdgeCaseType:
    def __init__(self, label, generator_func, min_count=10):
        self.label = label
        self.generator_func = generator_func
        self.min_count = min_count

    def generate(self):
        return self.generator_func(self.min_count)

class EdgeCaseGenerator:
    def __init__(self, inappropriate_count=50):
        self.inappropriate_count = inappropriate_count
        self.specials = ["!@#", "$%^", "&*()", "[]{}", ":;'<>,.?/|\\"]
        self.inappropriate_keywords = [
            "adult content", "explicit nude", "illegal drugs", "hate speech", "terrorist", "scam", "fake passport", "piracy", "violence"
        ]
        self.edge_case_types = [
            EdgeCaseType("very_short", self._generate_very_short),
            EdgeCaseType("very_long", self._generate_very_long),
            EdgeCaseType("numbers", self._generate_numbers),
            EdgeCaseType("empty", self._generate_empty),
            EdgeCaseType("gibberish", self._generate_gibberish),
            EdgeCaseType("ambiguous", self._generate_ambiguous),
            EdgeCaseType("special_chars", self._generate_special_chars),
        ]

    def _generate_very_short(self, count):
        return [
            {"business_description": chr(97 + i), "ideal_domain": f"{chr(97 + i)}.com", "label": "very_short"}
            for i in range(count)
        ]

    def _generate_very_long(self, count):
        long_words = [
            "Lopadotemachoselachogaleokranioleipsanodrimypotrimmatosilphioparaomelitokatakechymenokichlepikossyphophattoperisteralektryonoptekephalliokigklopeleiolagoiosiraiobaphetraganopterygon",
            "hippopotomonstrosesquipedaliophobia",
            "floccinaucinihilipilification",
            "antidisestablishmentarianism",
            "Supercalifragilisticexpialidocious",
            "Pneumonoultramicroscopicsilicovolcanoconiosis",
            "pseudopseudohypoparathyroidism",
            "spectrophotofluorometrically",
            "thyroparathyroidectomized",
            "dichlorodifluoromethane"
        ]
        return [
            {"business_description": f"{w} business with extremely long name for testing purposes only and more text to increase length", "ideal_domain": f"{w}.com", "label": "very_long"}
            for w in long_words[:count]
        ]

    def _generate_numbers(self, count):
        numbers_list = [str(n) for n in [2025, 7, 314, 123, 999, 42, 88, 56, 101, 77]]
        return [
            {"business_description": f"{n} Bakery", "ideal_domain": f"{n}bakery.com", "label": "numbers"}
            for n in numbers_list[:count]
        ]

    def _generate_empty(self, count):
        return [
            {"business_description": "", "ideal_domain": "", "label": "empty"} for _ in range(count)
        ]

    def _generate_gibberish(self, count):
        gibberish_list = [
            "qwertyuiop", "zxcvbnm", "asdfghjkl", "poiuytrewq", "mnbvcxz", "lkjhgfdsa", "gibberishtext", "nonsensewords", "blargh", "wibblewobble"
        ]
        return [
            {"business_description": s, "ideal_domain": "", "label": "gibberish"}
            for s in gibberish_list[:count]
        ]

    def _generate_ambiguous(self, count):
        ambiguous_list = [
            "???", "undefined", "random", "odd", "strange", "confusing", "mystery", "unknown", "enigmatic", "peculiar"
        ]
        return [
            {"business_description": s, "ideal_domain": "", "label": "ambiguous"}
            for s in ambiguous_list[:count]
        ]

    def _generate_special_chars(self, count):
        specials = self.specials * ((count // len(self.specials)) + 1)
        return [
            {"business_description": f"Cafe {s} with specials", "ideal_domain": "cafe-special.com", "label": "special_chars"}
            for s in specials[:count]
        ]

    def _generate_inappropriate(self):
        # Make inappropriate cases longer (between 40 and 120 chars)
        keywords = self.inappropriate_keywords * ((self.inappropriate_count // len(self.inappropriate_keywords)) + 1)
        import random
        cases = []
        for kw in keywords[:self.inappropriate_count]:
            # Generate a longer inappropriate business description
            desc = f"{kw} website "
            # Add more context to increase length
            while len(desc) < random.randint(40, 120):
                extra = random.choice([
                    "for adults only", "illegal content", "explicit material", "dangerous activity", "scam operation", "fake documents", "pirated media", "violent content", "hate group", "criminal network", "banned substances", "unlawful services", "prohibited items", "unsafe website", "malicious intent", "fraudulent business"
                ])
                desc += extra + ". "
            desc = desc.strip()
            cases.append({
                "business_description": desc,
                "ideal_domain": "",
                "label": "inappropriate"
            })
        return cases

    def generate(self):
        import random
        all_cases = []
        for edge_type in self.edge_case_types:
            all_cases.extend(edge_type.generate())
        all_cases.extend(self._generate_inappropriate())
        random.shuffle(all_cases)
        return pd.DataFrame(all_cases)

    def save(self, path="synthetic_domain_dataset_edge_cases.csv"):
        df = self.generate()
        df.to_csv(path, index=False)
        print(f"Edge/inappropriate cases saved to {path}")
