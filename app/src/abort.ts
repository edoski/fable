export function abortError(message: string): Error {
  const error = new Error(message);
  error.name = "AbortError";
  return error;
}
