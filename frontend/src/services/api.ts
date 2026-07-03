import axios from "axios";

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "/api/v1",
  timeout: 60000,
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      const data = error.response.data;
      const message = data?.message || error.message;
      return Promise.reject(new Error(message));
    }
    if (error.request) {
      return Promise.reject(new Error("No response from server"));
    }
    return Promise.reject(error);
  },
);

export default apiClient;
