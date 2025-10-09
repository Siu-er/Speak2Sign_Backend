import re
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class SiGMLGenerator:
    """Generator for SiGML (Signing Gesture Markup Language) from ASL gloss"""

    def __init__(self, signs_dir="data/signs"):
        self.signs_dir = signs_dir
        self.sigml_cache = {}
        self._load_sigml_files()

    def _load_sigml_files(self):
        """Load all SiGML files from the signs directory"""
        if not os.path.exists(self.signs_dir):
            logger.warning(f"Signs directory not found: {self.signs_dir}")
            return

        sigml_files = [f for f in os.listdir(self.signs_dir) if f.endswith('.sigml')]
        logger.info(f"Loading {len(sigml_files)} SiGML files from {self.signs_dir}")

        for filename in sigml_files:
            sign_name = os.path.splitext(filename)[0].upper()
            file_path = os.path.join(self.signs_dir, filename)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # Extract just the hns_sign content, not the full sigml wrapper
                    self.sigml_cache[sign_name] = self._extract_sign_content(content)

            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")

        logger.info(f"Successfully loaded {len(self.sigml_cache)} SiGML signs")

    def _extract_sign_content(self, sigml_content):
        """Extract the hns_sign content from full SiGML"""
        import re
        # Extract content between <hns_sign> tags
        match = re.search(r'<hns_sign[^>]*>(.*?)</hns_sign>', sigml_content, re.DOTALL)
        if match:
            return f'<hns_sign{match.group(0)[9:]}'  # Include the full hns_sign tag
        return sigml_content

    def gloss_to_tokens(self, gloss: str) -> List[str]:
        """Convert gloss string to list of tokens"""
        # Split on whitespace and forward slash, filter empty tokens
        tokens = [t for t in re.split(r"[\s/]+", gloss.strip()) if t]
        return tokens

    def generate_sigml_for_token(self, token: str) -> str:
        """Generate SiGML for a single token"""
        # Remove intensity markers (++)
        clean_token = token.replace("++", "")

        # Check if token is in loaded SiGML cache
        if clean_token.upper() in self.sigml_cache:
            sigml = self.sigml_cache[clean_token.upper()]

            # Add intensity if present
            if "++" in token:
                # For intensified signs, we can add duration or repetition
                sigml = sigml.replace("<hamhold/>", "<hamhold/><hamrepeatfromstart/>")

            return sigml.strip()

        # Handle fingerspelling
        if clean_token.startswith("FS-"):
            letters = clean_token[3:]  # Remove "FS-" prefix
            return self._generate_fingerspelling_from_files(letters)

        # Handle numbers
        if clean_token.isdigit():
            return self._generate_number_from_files(clean_token)

        # Handle compound signs with hyphens
        if "-" in clean_token and not clean_token.startswith("FS-"):
            parts = clean_token.split("-")
            if len(parts) == 2:
                return self._generate_compound_from_files(parts[0], parts[1])

        # Default fallback: fingerspell using actual letter files
        logger.info(f"No SiGML found for token: {clean_token}, using fingerspelling")
        return self._generate_fingerspelling_from_files(clean_token)

    def _generate_fingerspelling_from_files(self, text: str) -> str:
        """Generate SiGML for fingerspelling using actual letter files"""
        signs = []
        for letter in text.upper():
            if letter.isalpha() and letter in self.sigml_cache:
                signs.append(self.sigml_cache[letter])
            else:
                logger.warning(f"No SiGML file found for letter: {letter}")
                # Skip unknown characters
                continue

        if not signs:
            return self._generate_fallback_sign(f"FS-{text}")

        return "\n".join(signs)

    def _generate_number_from_files(self, number: str) -> str:
        """Generate SiGML for numbers using actual number files"""
        if number in self.sigml_cache:
            return self.sigml_cache[number]
        else:
            logger.warning(f"No SiGML file found for number: {number}")
            return self._generate_fallback_sign(number)

    def _generate_compound_from_files(self, part1: str, part2: str) -> str:
        """Generate SiGML for compound signs using actual files"""
        signs = []

        # Try to find SiGML for each part
        for part in [part1, part2]:
            if part.upper() in self.sigml_cache:
                signs.append(self.sigml_cache[part.upper()])
            elif part.isdigit() and part in self.sigml_cache:
                signs.append(self.sigml_cache[part])
            else:
                # Try fingerspelling the part
                signs.append(self._generate_fingerspelling_from_files(part))

        return "\n".join(signs)

    def _generate_fallback_sign(self, token: str) -> str:
        """Generate a minimal fallback sign for unknown tokens using available files"""
        # Try to use a neutral sign from available files
        neutral_signs = ['A', 'I', '0']  # Use basic signs as fallback

        for neutral in neutral_signs:
            if neutral in self.sigml_cache:
                # Modify the gloss to indicate it's a fallback
                fallback_sigml = self.sigml_cache[neutral]
                # Replace the gloss attribute to indicate the original token
                fallback_sigml = re.sub(r'gloss="[^"]*"', f'gloss="{token}"', fallback_sigml)
                return fallback_sigml

        # If no neutral signs available, generate minimal SiGML (last resort)
        return f"""<hns_sign gloss="{token}">
              <hamnosys_nonmanual/>
              <hamnosys_manual>
                <hamflathand/>
                <hamneutralspace/>
                <hamhold/>
              </hamnosys_manual>
            </hns_sign>"""

    def generate_sigml(self, gloss: str) -> str:
        """
        Generate complete SiGML document from ASL gloss

        Args:
            gloss: ASL gloss string (e.g., "HELLO FRIEND GOODBYE")

        Returns:
            Complete SiGML XML document
        """
        if not gloss or not gloss.strip():
            return self._create_empty_sigml()

        tokens = self.gloss_to_tokens(gloss)
        signs = []

        for token in tokens:
            try:
                sign_xml = self.generate_sigml_for_token(token)
                signs.append(sign_xml)
            except Exception as e:
                # Fallback to generic sign for problematic tokens
                print(f"Warning: Could not generate SiGML for token '{token}': {e}")
                signs.append(self._generate_fallback_sign(token))

        # Combine all signs into a complete SiGML document
        body = "\n".join(signs)

        return f"""<?xml version="1.0" encoding="UTF-8"?>
<sigml>
{body}
</sigml>"""

    def _create_empty_sigml(self) -> str:
        """Create an empty SiGML document"""
        return """<?xml version="1.0" encoding="UTF-8"?>
<sigml>
</sigml>"""

    def add_sign_to_cache(self, gloss: str, sigml: str):
        """Add a new sign to the SiGML cache"""
        self.sigml_cache[gloss.upper()] = sigml

    def get_available_signs(self) -> List[str]:
        """Get list of available signs in the cache"""
        return list(self.sigml_cache.keys())