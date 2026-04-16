"""
Tests for genomeid.py - Python port of Bio::Seq::GenomeID.pm

Each test validates a specific aspect of the ID generation to ensure
byte-for-byte compatibility with the Perl implementation.
"""

import base64
import os
import subprocess
import sys
import tempfile
import unittest

# Add repo root to path so we can import genomeid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import genomeid
from genomeid import (
    _pack_bits_to_b64,
    _unpack_b64_to_bits,
    _genMADIB,
    _guessHG,
    _State,
    _regen,
    _select_hg,
    _parse_vcf,
    _parse_tbi,
    generate_id,
    BIT_LOC_STATIC,
    PENGELLY_STATIC,
    AUTOSOMES_STATIC,
    ALLOSOMES_STATIC,
    REF_ALLEL_STATIC,
    VERSION,
)


# ---------------------------------------------------------------------------
# Helper: create a minimal bgzip+tabix-indexed VCF from lines
# ---------------------------------------------------------------------------
def _make_tbi(vcf_lines: list, dest_dir: str) -> str:
    """Write lines to a bgzipped+indexed VCF, return the .vcf.gz path."""
    try:
        import pysam  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("pysam not installed; skipping TBI tests")

    vcf_path = os.path.join(dest_dir, "test.vcf.gz")
    raw_path = os.path.join(dest_dir, "test.vcf")
    with open(raw_path, "w") as fh:
        fh.write("\n".join(vcf_lines) + "\n")
    # bgzip + tabix
    try:
        import pysam
        pysam.tabix_compress(raw_path, vcf_path, force=True)
        pysam.tabix_index(vcf_path, preset="vcf", force=True)
    except Exception as exc:
        raise unittest.SkipTest(f"Could not create tabix index: {exc}")
    return vcf_path


# ---------------------------------------------------------------------------
# 1. Bitstring packing and base64 encoding
# ---------------------------------------------------------------------------
class TestPackBits(unittest.TestCase):

    def test_all_zeros_72_bits(self):
        """72 zero bits → 9 zero bytes → 'AAAAAAAAAAAA'."""
        result = _pack_bits_to_b64("0" * 72)
        self.assertEqual(result, base64.b64encode(bytes(9)).decode())
        self.assertEqual(len(result), 12)

    def test_all_zeros_120_bits(self):
        """120 zero bits (with sex) → 15 zero bytes → 20 A's."""
        result = _pack_bits_to_b64("0" * 120)
        self.assertEqual(result, base64.b64encode(bytes(15)).decode())
        self.assertEqual(len(result), 20)

    def test_all_ones_72_bits(self):
        """72 one bits → 9 bytes of 0xFF → known base64."""
        result = _pack_bits_to_b64("1" * 72)
        expected = base64.b64encode(bytes([0xFF] * 9)).decode()
        self.assertEqual(result, expected)

    def test_no_padding_needed(self):
        """Bit strings whose length is already a multiple of 8 need no padding."""
        bits = "10101010" * 9   # 72 bits
        result = _pack_bits_to_b64(bits)
        b = int(bits, 2).to_bytes(9, "big")
        self.assertEqual(result, base64.b64encode(b).decode())

    def test_version_bits_in_gvb(self):
        """VERSION=1 encodes to '000001' (6 bits)."""
        gvb = format(VERSION, "06b")
        self.assertEqual(gvb, "000001")

    def test_roundtrip(self):
        """Pack then unpack gives back the original bits (padded to byte boundary)."""
        bits = "1100110011001100110011001100110011001100110011001100110011001100" \
               "100101001010010100101001010010100101001010010100101001010010101010101010101010101010"
        # Trim/pad to 120 bits (15 bytes)
        bits = bits[:120].ljust(120, "0")
        encoded = _pack_bits_to_b64(bits)
        decoded = _unpack_b64_to_bits(encoded)
        self.assertEqual(decoded, bits)


# ---------------------------------------------------------------------------
# 2. genMADIB
# ---------------------------------------------------------------------------
class TestGenMADIB(unittest.TestCase):

    def _make_state(self):
        st = _State()
        _regen(st)
        return st

    def test_zero_missing(self):
        st = self._make_state()
        _genMADIB(st, miss_count=0, only_peng=False, bit_mis=0)
        self.assertEqual(st.MADIB, "000000")

    def test_one_missing_bitmis_5(self):
        """1 missing: MADIB = 0-padded binary of bMis (bit index 5 → '000101')."""
        st = self._make_state()
        _genMADIB(st, miss_count=1, only_peng=False, bit_mis=5)
        self.assertEqual(st.MADIB, "000101")

    def test_one_missing_bitmis_0(self):
        st = self._make_state()
        _genMADIB(st, miss_count=1, only_peng=False, bit_mis=0)
        self.assertEqual(st.MADIB, "000000")

    def test_two_missing(self):
        st = self._make_state()
        _genMADIB(st, miss_count=2, only_peng=False, bit_mis=0)
        self.assertEqual(st.MADIB, "000010")

    def test_four_missing(self):
        st = self._make_state()
        _genMADIB(st, miss_count=4, only_peng=False, bit_mis=0)
        self.assertEqual(st.MADIB, "000100")

    def test_five_missing_encodes_63(self):
        """5 or more missing → 63 (binary '111111')."""
        st = self._make_state()
        _genMADIB(st, miss_count=5, only_peng=False, bit_mis=0)
        self.assertEqual(st.MADIB, "111111")
        # 63 in binary - Perl uses sprintf("%b",63) = "111111" with NO zero-padding
        self.assertEqual(len(st.MADIB), 6)

    def test_many_missing_encodes_63(self):
        st = self._make_state()
        _genMADIB(st, miss_count=58, only_peng=False, bit_mis=0)
        self.assertEqual(st.MADIB, "111111")

    def test_only_peng_encodes_59(self):
        """Pengelly-only mode → 59 (binary '111011')."""
        st = self._make_state()
        _genMADIB(st, miss_count=3, only_peng=True, bit_mis=0)
        self.assertEqual(st.MADIB, "111011")
        # 59 in binary - Perl uses sprintf("%b",59) = "111011" with NO zero-padding
        self.assertEqual(len(st.MADIB), 6)


# ---------------------------------------------------------------------------
# 3. Static dictionaries integrity
# ---------------------------------------------------------------------------
class TestStaticDicts(unittest.TestCase):

    def test_autosome_count(self):
        """Both hg19 and hg38 should have exactly 58 autosome markers."""
        for hg in ("hg19", "hg38"):
            autosome_keys = [k for k in BIT_LOC_STATIC[hg]
                             if not k.startswith("X:") and not k.startswith("Y:")]
            self.assertEqual(len(autosome_keys), 58, f"{hg} autosome count mismatch")

    def test_allosome_x_count(self):
        """Both hg19 and hg38 should have exactly 24 X chromosome markers."""
        for hg in ("hg19", "hg38"):
            x_keys = [k for k in BIT_LOC_STATIC[hg] if k.startswith("X:")]
            self.assertEqual(len(x_keys), 24, f"{hg} X marker count mismatch")

    def test_bit_locations_1_to_58(self):
        """Autosome bit values should cover 1-58 exactly."""
        for hg in ("hg19", "hg38"):
            auto_bits = sorted(
                int(BIT_LOC_STATIC[hg][k])
                for k in BIT_LOC_STATIC[hg]
                if not k.startswith("X:") and not k.startswith("Y:")
            )
            self.assertEqual(auto_bits, list(range(1, 59)),
                             f"{hg} bit values must be 1-58")

    def test_x_bit_locations_odd_1_to_47(self):
        """X chromosome bit values should be odd numbers 1-47."""
        for hg in ("hg19", "hg38"):
            x_bits = sorted(
                int(BIT_LOC_STATIC[hg][k])
                for k in BIT_LOC_STATIC[hg]
                if k.startswith("X:")
            )
            self.assertEqual(x_bits, list(range(1, 48, 2)),
                             f"{hg} X bit values must be 1,3,5,...,47")

    def test_pengelly_count(self):
        """pengelly_static has 58 entries per hg (one per autosome marker)."""
        for hg in ("hg19", "hg38"):
            self.assertEqual(len(PENGELLY_STATIC[hg]), 58)

    def test_autosomes_autoSomes_keys_match(self):
        """autoSomes_static keys should match autosome keys in bit_loc_static."""
        for hg in ("hg19", "hg38"):
            auto_keys_bitloc = {k for k in BIT_LOC_STATIC[hg]
                                if not k.startswith("X:") and not k.startswith("Y:")}
            auto_keys_autosomes = set(AUTOSOMES_STATIC[hg].keys())
            self.assertEqual(auto_keys_bitloc, auto_keys_autosomes,
                             f"{hg} autoSomes/bit_loc key mismatch")

    def test_y_markers_have_bit_depth_tree(self):
        """All Y markers in bit_loc_static should have 'bit:depth:marker' format."""
        for hg in ("hg19", "hg38"):
            for key, val in BIT_LOC_STATIC[hg].items():
                if key.startswith("Y:"):
                    self.assertIsInstance(val, str, f"{hg} {key} val should be str")
                    parts = val.split(":")
                    self.assertGreaterEqual(len(parts), 3,
                                           f"{hg} {key} val should be bit:depth:tree")

    def test_hg19_sample_marker(self):
        """Spot-check hg19 autosome marker '1:45973928' → bit 1."""
        self.assertEqual(BIT_LOC_STATIC["hg19"]["1:45973928"], 1)

    def test_hg38_sample_marker(self):
        """Spot-check hg38 autosome marker '1:45508256' → bit 1."""
        self.assertEqual(BIT_LOC_STATIC["hg38"]["1:45508256"], 1)


# ---------------------------------------------------------------------------
# 4. guessHG switching
# ---------------------------------------------------------------------------
class TestGuessHG(unittest.TestCase):

    def _make_state(self, hg: str = "hg38") -> _State:
        st = _State()
        _regen(st)
        _select_hg(st, hg)
        st.guess_hg = True
        return st

    def test_unknown_key_returns_true(self):
        """Key not in any build should return True (skip) without dying."""
        st = self._make_state("hg38")
        result = _guessHG(st, key="99:12345678", smp_ref="A")
        self.assertTrue(result)
        self.assertEqual(st.ref_hg, "hg38")  # unchanged

    def test_key_matches_current_build(self):
        """When ref allele matches current build exactly, stay and return True."""
        st = self._make_state("hg38")
        # Pick a key in hg38 ref_allel that isn't in hg19
        # and whose ref allele in hg38 matches
        key = "1:45508256"  # hg38 autosome
        ref_in_hg38 = REF_ALLEL_STATIC["hg38"].get(key)
        if ref_in_hg38 is None:
            self.skipTest("Key not in hg38 ref_allel_static")
        result = _guessHG(st, key=key, smp_ref=ref_in_hg38)
        self.assertTrue(result)

    def test_switches_to_hg19(self):
        """When ref allele matches hg19 but not hg38, switch to hg19."""
        st = self._make_state("hg38")
        # Find a key that:
        #  - exists in both hg19 and hg38 ref_allel_static
        #  - has DIFFERENT ref alleles in the two builds
        for key in REF_ALLEL_STATIC["hg38"]:
            ref38 = REF_ALLEL_STATIC["hg38"].get(key, "")
            ref19 = REF_ALLEL_STATIC["hg19"].get(key, "")
            if ref38 and ref19 and ref38 != ref19:
                result = _guessHG(st, key=key, smp_ref=ref19)
                self.assertFalse(result)
                self.assertEqual(st.ref_hg, "hg19")
                return
        self.skipTest("No key with differing ref alleles found")


# ---------------------------------------------------------------------------
# 5. VCF parsing end-to-end
# ---------------------------------------------------------------------------
class TestVCFParsing(unittest.TestCase):

    def _write_vcf(self, lines: list) -> str:
        """Write lines to a temp VCF, return path."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False)
        f.write("\n".join(lines) + "\n")
        f.flush()
        f.close()
        return f.name

    def tearDown(self):
        # Clean up temp files created during tests (best effort)
        for attr in ("_tmpfile",):
            path = getattr(self, attr, None)
            if path and os.path.exists(path):
                os.unlink(path)

    def test_empty_vcf_no_sex(self):
        """Empty VCF → all 58 markers missing → MADIB='111111', AMB all 0."""
        path = self._write_vcf([
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
        ])
        self._tmpfile = path
        gid = generate_id(file=path, type="vcf", hg="hg38", sex=False)
        bits = _unpack_b64_to_bits(gid)
        madib = bits[:6]
        self.assertEqual(madib, "111111",
                         "All markers missing → MADIB should be 111111")
        smb = bits[64:66]
        self.assertEqual(smb[1], "1", "SMB[1] should be 1 when no Y marker")
        self.assertEqual(len(gid), 12, "Without sex: ID length should be 12")

    def test_one_het_marker_hg38(self):
        """One heterozygous autosome marker → AMB[0]=1, 57 missing."""
        # hg38 bit-1 marker: 1:45508256
        path = self._write_vcf([
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
            "1\t45508256\t.\tA\tG\t.\t.\t.\tGT\t0/1",
        ])
        self._tmpfile = path
        gid = generate_id(file=path, type="vcf", hg="hg38", sex=False)
        bits = _unpack_b64_to_bits(gid)
        # AMB starts at bit 6; bit-1 marker is AMB[0] → bit index 6
        self.assertEqual(bits[6], "1", "AMB[0] should be 1 for het marker")
        # 57 markers still missing → miss_count = 58-1 = 57 > 4 → MADIB = '111111'
        self.assertEqual(bits[:6], "111111")

    def test_one_homo_ref_marker_hg38(self):
        """One homozygous ref marker → AMB[0]=0, 57 missing."""
        path = self._write_vcf([
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
            "1\t45508256\t.\tA\tG\t.\t.\t.\tGT\t0/0",
        ])
        self._tmpfile = path
        gid = generate_id(file=path, type="vcf", hg="hg38", sex=False)
        bits = _unpack_b64_to_bits(gid)
        self.assertEqual(bits[6], "0", "AMB[0] should be 0 for homo-ref marker")

    def test_one_homo_alt_marker_hg38(self):
        """One homozygous alt marker → AMB[0]=0 (homo), 57 missing."""
        path = self._write_vcf([
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
            "1\t45508256\t.\tA\tG\t.\t.\t.\tGT\t1/1",
        ])
        self._tmpfile = path
        gid = generate_id(file=path, type="vcf", hg="hg38", sex=False)
        bits = _unpack_b64_to_bits(gid)
        self.assertEqual(bits[6], "0", "AMB[0] should be 0 for homo-alt marker")

    def test_missing_marker_gt(self):
        """Marker with ./. genotype counts as missing."""
        path = self._write_vcf([
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
            "1\t45508256\t.\tA\tG\t.\t.\t.\tGT\t./.",
        ])
        self._tmpfile = path
        gid = generate_id(file=path, type="vcf", hg="hg38", sex=False)
        bits = _unpack_b64_to_bits(gid)
        # Missing marker, all 58 missing → MADIB = '111111'
        self.assertEqual(bits[:6], "111111")

    def test_chr_prefix_stripped(self):
        """VCF with 'chr' prefix should give same result as without."""
        path_no_chr = self._write_vcf([
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
            "1\t45508256\t.\tA\tG\t.\t.\t.\tGT\t0/1",
        ])
        path_chr = self._write_vcf([
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
            "chr1\t45508256\t.\tA\tG\t.\t.\t.\tGT\t0/1",
        ])
        gid_no_chr = generate_id(file=path_no_chr, type="vcf", hg="hg38", sex=False)
        gid_chr    = generate_id(file=path_chr,    type="vcf", hg="hg38", sex=False)
        os.unlink(path_no_chr)
        os.unlink(path_chr)
        self.assertEqual(gid_no_chr, gid_chr,
                         "chr prefix in VCF should not affect output")

    def test_gvb_encodes_version_1(self):
        """GVB bits (66-71) should encode VERSION=1 as '000001'."""
        path = self._write_vcf([
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
        ])
        self._tmpfile = path
        gid = generate_id(file=path, type="vcf", hg="hg38", sex=False)
        bits = _unpack_b64_to_bits(gid)
        gvb = bits[66:72]
        self.assertEqual(gvb, "000001", f"GVB should be '000001' but got '{gvb}'")

    def test_sex_flag_extends_id_to_20_chars(self):
        """With sex=True, ID length should be 20 chars (120 bits)."""
        path = self._write_vcf([
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
        ])
        self._tmpfile = path
        gid = generate_id(file=path, type="vcf", hg="hg38", sex=True)
        self.assertEqual(len(gid), 20, f"With sex=True: ID length should be 20 but got {len(gid)}")

    def test_xmb_undefined_state_when_no_sex_markers(self):
        """Without X/Y markers, XMB should be in 'undefined' state (pairs '10')."""
        path = self._write_vcf([
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
        ])
        self._tmpfile = path
        gid = generate_id(file=path, type="vcf", hg="hg38", sex=True)
        bits = _unpack_b64_to_bits(gid)
        xmb = bits[72:]  # 48 bits
        # When no sex markers are seen, every pair should be '10' (undefined)
        # NOTE: with no Y detected and sex=True, XMB stays as initialized ('10' pairs)
        for i in range(0, 48, 2):
            pair = xmb[i:i+2]
            self.assertEqual(pair, "10",
                             f"XMB pair at index {i} should be '10' (undefined), got '{pair}'")

    def test_multi_sample_vcf_default_column(self):
        """Without sampleName, the first sample column (col 9) is used."""
        path = self._write_vcf([
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2",
            "1\t45508256\t.\tA\tG\t.\t.\t.\tGT\t0/1\t0/0",
        ])
        self._tmpfile = path
        gid = generate_id(file=path, type="vcf", hg="hg38", sex=False)
        bits = _unpack_b64_to_bits(gid)
        # Default column is 9 (S1), which is het → AMB[0]=1
        self.assertEqual(bits[6], "1", "Default sample col 9 should be used (S1=0/1)")

    def test_multi_sample_vcf_sample_name_selection(self):
        """With sampleName='S2', the second sample column (col 10) is used."""
        path = self._write_vcf([
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2",
            "1\t45508256\t.\tA\tG\t.\t.\t.\tGT\t0/1\t0/0",
        ])
        self._tmpfile = path
        gid = generate_id(file=path, type="vcf", hg="hg38", sex=False,
                          sampleName="S2")
        bits = _unpack_b64_to_bits(gid)
        # S2 is 0/0 → homo → AMB[0]=0
        self.assertEqual(bits[6], "0", "S2 sample col 10 should be used (S2=0/0)")

    def test_all_markers_het_no_missing(self):
        """All 58 autosome markers present + het → MADIB='000000', AMB all 1."""
        auto_keys_hg38 = sorted(
            (k for k in BIT_LOC_STATIC["hg38"]
             if not k.startswith("X:") and not k.startswith("Y:")),
            key=lambda k: (int(k.split(":")[0]) if k.split(":")[0].isdigit() else 99,
                           int(k.split(":")[1]))
        )
        vcf_lines = [
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
        ]
        for key in auto_keys_hg38:
            chrom, pos = key.split(":")
            anc_alt = AUTOSOMES_STATIC["hg38"][key].split(":")
            ref_base = anc_alt[0]
            alt_base = anc_alt[1]
            vcf_lines.append(
                f"{chrom}\t{pos}\t.\t{ref_base}\t{alt_base}\t.\t.\t.\tGT\t0/1"
            )

        path = self._write_vcf(vcf_lines)
        self._tmpfile = path
        gid = generate_id(file=path, type="vcf", hg="hg38", sex=False)
        bits = _unpack_b64_to_bits(gid)
        # All 58 markers present and het → AMB all 1s
        amb = bits[6:64]
        self.assertEqual(amb, "1" * 58, "All AMB bits should be 1 (all het)")
        # 0 missing → MADIB = '000000'
        madib = bits[:6]
        self.assertEqual(madib, "000000")


# ---------------------------------------------------------------------------
# 6. TBI (tabix) parsing
# ---------------------------------------------------------------------------
class TestTBIParsing(unittest.TestCase):

    def test_empty_tbi_no_sex(self):
        """Tabix VCF with no matching records → all missing → MADIB='111111'."""
        lines = [
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
            "22\t99999999\t.\tA\tG\t.\t.\t.\tGT\t0/1",  # non-marker position
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                vcf_path = _make_tbi(lines, tmpdir)
            except unittest.SkipTest as e:
                self.skipTest(str(e))

            gid = generate_id(file=vcf_path, type="tbi", hg="hg38", sex=False)
            bits = _unpack_b64_to_bits(gid)
            self.assertEqual(bits[:6], "111111",
                             "All markers missing → MADIB should be 111111")
            self.assertEqual(len(gid), 12)

    def test_one_het_marker_tbi(self):
        """Tabix VCF with one het marker → AMB[0]=1."""
        lines = [
            "##fileformat=VCFv4.1",
            "#CHROM\t POS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
            "1\t45508256\t.\tA\tG\t.\t.\t.\tGT\t0/1",
        ]
        # Remove the duplicate header line
        lines = [
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
            "1\t45508256\t.\tA\tG\t.\t.\t.\tGT\t0/1",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                vcf_path = _make_tbi(lines, tmpdir)
            except unittest.SkipTest as e:
                self.skipTest(str(e))

            gid = generate_id(file=vcf_path, type="tbi", hg="hg38", sex=False)
            bits = _unpack_b64_to_bits(gid)
            self.assertEqual(bits[6], "1",
                             "AMB[0] should be 1 for het marker at 1:45508256")

    def test_tbi_with_chr_prefix(self):
        """Tabix VCF with chr-prefixed chromosomes should work."""
        lines = [
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
            "chr1\t45508256\t.\tA\tG\t.\t.\t.\tGT\t0/1",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                vcf_path = _make_tbi(lines, tmpdir)
            except unittest.SkipTest as e:
                self.skipTest(str(e))

            gid = generate_id(file=vcf_path, type="tbi", hg="hg38", sex=False)
            bits = _unpack_b64_to_bits(gid)
            # Should find the marker via chr-prefix fallback
            self.assertEqual(bits[6], "1",
                             "chr-prefixed TBI should find 1:45508256 marker")


# ---------------------------------------------------------------------------
# 7. XMB builder (allosomal VCF)
# ---------------------------------------------------------------------------
class TestXMBBuilder(unittest.TestCase):

    def _vcf_with_x_marker(self, chrom: str, pos: str, gt: str) -> str:
        """Create VCF with one X marker, return path."""
        lines = [
            "##fileformat=VCFv4.1",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
            f"{chrom}\t{pos}\t.\tG\tA\t.\t.\t.\tGT\t{gt}",
        ]
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False)
        f.write("\n".join(lines) + "\n")
        f.close()
        return f.name

    def test_het_x_marker(self):
        """Het X marker (0/1) → XMB[bit]='0', XMB[bit+1]='1'."""
        # hg38 X:4148702 → bit_loc[X:4148702] = 1 → XMB index 0
        path = self._vcf_with_x_marker("X", "4148702", "0/1")
        try:
            gid = generate_id(file=path, type="vcf", hg="hg38", sex=True)
        finally:
            os.unlink(path)
        bits = _unpack_b64_to_bits(gid)
        xmb = bits[72:]
        # bit = int(1) - 1 = 0 → XMB[0:2] = "01"
        self.assertEqual(xmb[:2], "01",
                         f"Het X marker: XMB[0:2] should be '01' but got '{xmb[:2]}'")

    def test_homo_ref_x_marker(self):
        """Homo-ref X marker (0/0) → XMB[bit]='0', XMB[bit+1]='0'."""
        # hg38 X:4148702 → bit 1 → XMB[0:2]
        # REF = G = anc (allosomes[X:4148702] = 'G:A', anc='G')
        path = self._vcf_with_x_marker("X", "4148702", "0/0")
        try:
            gid = generate_id(file=path, type="vcf", hg="hg38", sex=True)
        finally:
            os.unlink(path)
        bits = _unpack_b64_to_bits(gid)
        xmb = bits[72:]
        self.assertEqual(xmb[:2], "00",
                         f"Homo-ref X: XMB[0:2] should be '00' but got '{xmb[:2]}'")

    def test_missing_x_marker_stays_undefined(self):
        """Missing X marker (./.) → XMB pair stays '10' (undefined)."""
        path = self._vcf_with_x_marker("X", "4148702", "./.")
        try:
            gid = generate_id(file=path, type="vcf", hg="hg38", sex=True)
        finally:
            os.unlink(path)
        bits = _unpack_b64_to_bits(gid)
        xmb = bits[72:]
        self.assertEqual(xmb[:2], "10",
                         f"Missing X: XMB[0:2] should stay '10' but got '{xmb[:2]}'")


# ---------------------------------------------------------------------------
# 8. Bit-layout integration: MADIB + AMB + SMB + GVB positions
# ---------------------------------------------------------------------------
class TestBitLayout(unittest.TestCase):

    def _gid_bits(self, **kwargs) -> str:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vcf", delete=False) as f:
            f.write("##fileformat=VCFv4.1\n")
            f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n")
            path = f.name
        try:
            gid = generate_id(file=path, **kwargs)
        finally:
            os.unlink(path)
        return _unpack_b64_to_bits(gid)

    def test_layout_without_sex(self):
        """Without sex: total 72 bits; MADIB(0-5) AMB(6-63) SMB(64-65) GVB(66-71)."""
        bits = self._gid_bits(type="vcf", hg="hg38", sex=False)
        self.assertEqual(len(bits), 72)

    def test_layout_with_sex(self):
        """With sex: total 120 bits; XMB starts at bit 72."""
        bits = self._gid_bits(type="vcf", hg="hg38", sex=True)
        self.assertEqual(len(bits), 120)

    def test_smb_position(self):
        """SMB occupies bits 64-65."""
        bits = self._gid_bits(type="vcf", hg="hg38", sex=False)
        smb = bits[64:66]
        # No Y → SMB[0]='0'; ucn=0, SMB[0]≠'1' → SMB[1]='1'
        self.assertEqual(smb, "01")

    def test_gvb_position(self):
        """GVB occupies bits 66-71 and encodes VERSION=1."""
        bits = self._gid_bits(type="vcf", hg="hg38", sex=False)
        gvb = bits[66:72]
        self.assertEqual(gvb, "000001")


if __name__ == "__main__":
    unittest.main(verbosity=2)
