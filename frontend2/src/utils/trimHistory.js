// import { TrimHistoryItem } from '../types'; // Changed from ../types.js

/**
 * Merge existing trim regions with new ones, avoiding duplicates and merging overlaps.
 * @param existingItems - Array of trim items
 * @param newItems - Array of new trim items
 * @returns Merged and sorted array of trim items, ensuring timestamp is present.
 */
export function mergeTrimItems(
  existingItems, 
  newItems
) {
  const all= [...existingItems];
  
  newItems.forEach(newItem => {
    // Ensure newItem has a timestamp; if not, it might indicate an issue upstream
    // or this function needs to assign one to items that lack it.
    // For now, assuming newItems are well-formed TrimHistoryItem from types.ts.
    const isDuplicate = all.some(existingItem =>
      existingItem.id === newItem.id ||
      (Math.abs(existingItem.start - newItem.start) < 0.001 && Math.abs(existingItem.end - newItem.end) < 0.001)
    );
    if (!isDuplicate) {
      all.push(newItem); // newItem should already conform to TrimHistoryItem including a non-optional timestamp
    }
  });

  all.sort((a, b) => a.start - b.start);

  if (all.length === 0) return [];

  const merged = [];
  // Initialize with the first item, ensuring it conforms. Spread copies all properties.
  // If all[0] comes from existingItems, it should already have a non-optional timestamp.
  merged.push({ ...all[0] }); 

  for (let i = 1; i < all.length; i++) {
    const currentItem = all[i]; // Should also have a non-optional timestamp
    const lastMergedItem = merged[merged.length - 1];

    if (currentItem.start <= lastMergedItem.end) { 
      lastMergedItem.end = Math.max(lastMergedItem.end, currentItem.end);
      lastMergedItem.id = `merged-${lastMergedItem.start.toFixed(3)}-${lastMergedItem.end.toFixed(3)}`;
      // lastMergedItem.timestamp could be updated if needed, e.g., to currentItem.timestamp or new Date().toISOString()
      // For now, it retains the timestamp of the item it was initialized from or extended into.
      // The shared TrimHistoryItem defines timestamp as required string. This means lastMergedItem.timestamp must be valid.
    } else {
      merged.push({ ...currentItem });
    }
  }
  return merged;
}