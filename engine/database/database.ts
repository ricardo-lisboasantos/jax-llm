/**
 * Defines a generic interface for key-value storage.
 */
export interface Database<K, V> {
  /** Retrieves a value by key. */
  get(key: K): Promise<V>;
  /** Adds a new key-value pair. */
  add(key: K, value: V): Promise<void>;
  /** Updates an existing key-value pair. */
  update(key: K, new_value: V): Promise<void>;
}
