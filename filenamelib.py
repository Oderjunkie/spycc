from contextlib import contextmanager
from os import unlink

@contextmanager
def temporary_filename(suffix=None):
  """Context that introduces a temporary file.

  Creates a temporary file, yields its name, and upon context exit, deletes it.
  (In contrast, tempfile.NamedTemporaryFile() provides a 'file' object and
  deletes the file as soon as that file object is closed, so the temporary file
  cannot be safely re-opened by another library or process.)

  Args:
    suffix: desired filename extension (e.g. '.mp4').

  Yields:
    The name of the temporary file.
  """
  import tempfile
  try:
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_name = f.name
    f.close()
    yield tmp_name
  finally:
    unlink(tmp_name)