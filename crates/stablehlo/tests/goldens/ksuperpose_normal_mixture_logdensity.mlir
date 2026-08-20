module {
  func.func @logdensity() -> tensor<f32> {
    %0 = stablehlo.constant dense<0.3> : tensor<f32>
    %1 = stablehlo.constant dense<1.2> : tensor<f32>
    %2 = stablehlo.reshape %0 : (tensor<f32>) -> tensor<1xf32>
    %3 = stablehlo.reshape %1 : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %5 = stablehlo.log %4 : tensor<2xf32>
    %6 = stablehlo.constant dense<0.5> : tensor<f32>
    %7 = stablehlo.constant dense<-1.0> : tensor<f32>
    %8 = stablehlo.constant dense<2.0> : tensor<f32>
    %9 = stablehlo.reshape %7 : (tensor<f32>) -> tensor<1xf32>
    %10 = stablehlo.reshape %8 : (tensor<f32>) -> tensor<1xf32>
    %11 = stablehlo.concatenate %9, %10, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %12 = stablehlo.constant dense<1.0> : tensor<f32>
    %13 = stablehlo.constant dense<0.5> : tensor<f32>
    %14 = stablehlo.reshape %12 : (tensor<f32>) -> tensor<1xf32>
    %15 = stablehlo.reshape %13 : (tensor<f32>) -> tensor<1xf32>
    %16 = stablehlo.concatenate %14, %15, dim = 0 : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
    %17 = stablehlo.log %16 : tensor<2xf32>
    %18 = stablehlo.negate %17 : tensor<2xf32>
    %19 = stablehlo.constant dense<-0.9189385332046727> : tensor<f32>
    %20 = stablehlo.broadcast_in_dim %6, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %21 = stablehlo.subtract %20, %11 : tensor<2xf32>
    %22 = stablehlo.divide %21, %16 : tensor<2xf32>
    %23 = stablehlo.constant dense<-0.5> : tensor<f32>
    %24 = stablehlo.multiply %22, %22 : tensor<2xf32>
    %25 = stablehlo.broadcast_in_dim %23, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %26 = stablehlo.multiply %25, %24 : tensor<2xf32>
    %27 = stablehlo.broadcast_in_dim %19, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %28 = stablehlo.add %18, %27 : tensor<2xf32>
    %29 = stablehlo.add %28, %26 : tensor<2xf32>
    %30 = stablehlo.add %5, %29 : tensor<2xf32>
    %31 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %32 = stablehlo.reduce(%30 init: %31) applies stablehlo.maximum across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %33 = stablehlo.broadcast_in_dim %32, dims = [] : (tensor<f32>) -> tensor<2xf32>
    %34 = stablehlo.subtract %30, %33 : tensor<2xf32>
    %35 = stablehlo.exponential %34 : tensor<2xf32>
    %36 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %37 = stablehlo.reduce(%35 init: %36) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %38 = stablehlo.log %37 : tensor<f32>
    %39 = stablehlo.add %38, %32 : tensor<f32>
    %40 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %41 = stablehlo.reduce(%4 init: %40) applies stablehlo.add across dimensions = [0] : (tensor<2xf32>, tensor<f32>) -> tensor<f32>
    %42 = stablehlo.log %41 : tensor<f32>
    %43 = stablehlo.subtract %39, %42 : tensor<f32>
    return %43 : tensor<f32>
  }
}
