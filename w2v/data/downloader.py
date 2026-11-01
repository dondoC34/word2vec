import logging
from pathlib import Path

logger = logging.getLogger("DataDownloader")


def ensure_nltk_gutenberg_corpora(
    target_dir: str | Path = "./corpuses",
    target_corpora: list[str] | None = None,
) -> list[Path]:
    """
    Data ingestion and synchronization utility:
    Ensures that a selection of classic literary works from NLTK Gutenberg
    are available on disk in `target_dir`.

    Calculates the delta against existing files in the directory so that
    pre-existing custom files (such as 'portrait_of_dorian_gray.txt') are
    preserved and only missing corpora are downloaded from NLTK.

    Args:
        target_dir: Destination directory where corpus files reside.
        target_corpora: Optional custom list of NLTK Gutenberg book identifiers.
                        Defaults to a curated selection of 18 classic works.

    Returns:
        List of Path objects for newly downloaded delta files.
    """
    dest = Path(target_dir)
    dest.mkdir(parents=True, exist_ok=True)

    if target_corpora is None:
        target_corpora = [
            "carroll-alice.txt",
            "melville-moby_dick.txt",
            "austen-emma.txt",
            "austen-persuasion.txt",
            "austen-sense.txt",
            "chesterton-thursday.txt",
            "chesterton-brown.txt",
            "chesterton-ball.txt",
            "shakespeare-hamlet.txt",
            "shakespeare-macbeth.txt",
            "shakespeare-caesar.txt",
            "milton-paradise.txt",
            "blake-poems.txt",
            "whitman-leaves.txt",
            "bryant-stories.txt",
            "burgess-busterbrown.txt",
            "edgeworth-parents.txt",
            "bible-kjv.txt",
        ]

    existing_files = {f.name.lower() for f in dest.iterdir() if f.is_file()}
    missing = [fname for fname in target_corpora if fname.lower() not in existing_files]

    logger.info("Scanning directory '%s' for NLTK Gutenberg corpora...", target_dir)
    logger.info("Found %d pre-existing file(s) in folder (preserved).", len(existing_files))

    if not missing:
        logger.info("All %d target NLTK corpora are already present on disk. No download needed.", len(target_corpora))
        return []

    logger.info("Identified delta of %d missing corpus file(s) to fetch from NLTK: %s", len(missing), missing)

    try:
        import nltk

        nltk.download("gutenberg", quiet=True)
        from nltk.corpus import gutenberg
    except Exception as e:
        logger.error("Failed to import or download NLTK gutenberg package: %s", e)
        return []

    downloaded: list[Path] = []
    for fname in missing:
        try:
            raw_text = gutenberg.raw(fname)
            out_file = dest / fname
            with open(out_file, "w", encoding="utf-8") as fp:
                fp.write(raw_text)
            logger.info("Downloaded delta corpus file: '%s' (%d characters).", fname, len(raw_text))
            downloaded.append(out_file)
        except Exception as e:
            logger.error("Failed to write corpus file '%s': %s", fname, e)

    logger.info(
        "Delta synchronization finished: %d new file(s) successfully written to '%s'.", len(downloaded), target_dir
    )
    return downloaded
