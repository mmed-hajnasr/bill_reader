import unittest
import core as mine
import re 

def isprice(text):
    text.strip()
    pattern = re.compile("^[£€\$]?\d+[,\.]\d+$|^\d+[,\.]\d+[£€\$]?$")
    if pattern.match(text):
        return not mine.math.isclose(mine.strip_price(text),0, rel_tol=1e-07, abs_tol=0.0)
    return False

class TestIsPrice(unittest.TestCase):

    def test_valid_price(self):
        self.assertTrue(isprice("10.99gfd"))
        self.assertTrue(isprice("1.5$"))
        self.assertTrue(isprice("0.50"))
        self.assertTrue(isprice("$1000,00"))
        self.assertTrue(isprice("1000,00€"))

    def test_invalid_price(self):
        self.assertFalse(isprice("$00000.000"))
        self.assertFalse(isprice("$1.thf"))
        self.assertFalse(isprice("1,000.00"))
        self.assertFalse(isprice("100€0"))
        self.assertFalse(isprice("1000"))
        self.assertFalse(isprice("$10.00$"))

if __name__ == '__main__':
    unittest.main()
# [£€\$]