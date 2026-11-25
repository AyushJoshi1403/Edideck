from edideck import core


def words(n=60, step=0.5):
    return [core.Word(f"w{i}" + ("." if i % 12 == 11 else ""), i * step, i * step + 0.4) for i in range(n)]


def test_pick_respects_length_and_overlap():
    clips = core.pick(words(), n=2, lo=5, hi=12)
    assert clips and all(5 <= c.end - c.start <= 12 for c in clips)
    assert all(a.end <= b.start or b.end <= a.start for a in clips for b in clips if a is not b)


def test_ass_has_karaoke_tags():
    out = core.ass(words(8), 0, 4)
    assert "\\k" in out and "Dialogue:" in out


def test_crop_expression_follows_track():
    assert core._crop_x([]) == "(iw-ow)/2"
    assert "if(lt(t" in core._crop_x([[0, 0.3], [1.5, 0.6]])


def test_styles_change_output():
    ws = words(8)
    assert "W0" in core.ass(ws, 0, 4, "pop") and "w0" in core.ass(ws, 0, 4, "bold")
