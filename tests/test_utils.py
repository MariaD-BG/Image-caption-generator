"""
tests for utils
"""
import matplotlib

from ICGmodel.utils import strip_syntax, plot_loss

matplotlib.use('Agg') # no matplotlib windows popping out

def test_strip_syntax_1():
    """
    Test that logic removes numbers and punctuation.
    """
    raw_input = "Hello, World! 123"

    expected_output = "Hello World "

    result = strip_syntax(raw_input)
    assert result == expected_output

def test_strip_syntax_2():
    """
    verify edge cases
    """

    assert strip_syntax("") == ""

    assert strip_syntax("@#$%^&*()") == ""

    assert strip_syntax("PyThOn") == "PyThOn"



def test_plot_loss(tmp_path):
    """
    Real test: Runs the actual plotting code and checks if a PNG is saved.
    We use tmp_path so we don't clutter the actual file system.
    """

    train_loss = [0.9, 0.8, 0.7, 0.6, 0.5]

    val_loss = [0.85, 0.65]


    save_file = tmp_path / "test_plot.png"

    plot_loss(
        train_loss=train_loss,
        val_loss=val_loss,
        save_path=str(save_file),
        val_interval=2
    )

    assert save_file.exists(), "The plot file was not created"

    assert save_file.stat().st_size > 0, "The plot file is empty"
