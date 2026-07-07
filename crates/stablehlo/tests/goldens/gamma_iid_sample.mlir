module {
  func.func @sample(%key: tensor<2xui64>) -> (tensor<4xf32>, tensor<2xui64>) {
    %0 = stablehlo.constant dense<2.0> : tensor<f32>
    %1 = stablehlo.constant dense<1.0> : tensor<f32>
    %2 = stablehlo.constant dense<0.0> : tensor<f32>
    %3 = stablehlo.constant dense<1.0> : tensor<f32>
    %4 = stablehlo.compare LT, %0, %3 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %5 = stablehlo.add %0, %3 : tensor<f32>
    %6 = stablehlo.select %4, %5, %0 : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = stablehlo.constant dense<0.3333333333333333> : tensor<f32>
    %8 = stablehlo.subtract %6, %7 : tensor<f32>
    %9 = stablehlo.constant dense<9.0> : tensor<f32>
    %10 = stablehlo.multiply %9, %8 : tensor<f32>
    %11 = stablehlo.sqrt %10 : tensor<f32>
    %12 = stablehlo.divide %3, %11 : tensor<f32>
    %13, %14 = stablehlo.rng_bit_generator %key, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %15 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %16 = stablehlo.shift_right_logical %14, %15 : tensor<128x4xui32>
    %17 = stablehlo.convert %16 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %18 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %19 = stablehlo.multiply %17, %18 : tensor<128x4xf32>
    %20 = stablehlo.constant dense<2.0> : tensor<128x4xf32>
    %21 = stablehlo.constant dense<1.0> : tensor<128x4xf32>
    %22 = stablehlo.multiply %19, %20 : tensor<128x4xf32>
    %23 = stablehlo.subtract %22, %21 : tensor<128x4xf32>
    %24 = chlo.erf_inv %23 : tensor<128x4xf32> -> tensor<128x4xf32>
    %25 = stablehlo.constant dense<1.4142135> : tensor<128x4xf32>
    %26 = stablehlo.multiply %24, %25 : tensor<128x4xf32>
    %27 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<128x4xf32>
    %28 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<128x4xf32>
    %29 = stablehlo.multiply %26, %27 : tensor<128x4xf32>
    %30 = stablehlo.add %29, %28 : tensor<128x4xf32>
    %31, %32 = stablehlo.rng_bit_generator %13, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<128x4xui32>)
    %33 = stablehlo.constant dense<9> : tensor<128x4xui32>
    %34 = stablehlo.shift_right_logical %32, %33 : tensor<128x4xui32>
    %35 = stablehlo.convert %34 : (tensor<128x4xui32>) -> tensor<128x4xf32>
    %36 = stablehlo.constant dense<1.1920929E-7> : tensor<128x4xf32>
    %37 = stablehlo.multiply %35, %36 : tensor<128x4xf32>
    %38 = stablehlo.subtract %3, %2 : tensor<f32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [] : (tensor<f32>) -> tensor<128x4xf32>
    %40 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<128x4xf32>
    %41 = stablehlo.multiply %37, %39 : tensor<128x4xf32>
    %42 = stablehlo.add %41, %40 : tensor<128x4xf32>
    %43 = stablehlo.constant dense<0> : tensor<i32>
    %44 = stablehlo.constant dense<false> : tensor<4xi1>
    %45 = stablehlo.constant dense<0.0> : tensor<4xf32>
    %49:3 = stablehlo.while(%46 = %43, %47 = %44, %48 = %45) : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    cond {
      %50 = stablehlo.constant dense<128> : tensor<i32>
      %51 = stablehlo.compare LT, %46, %50, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
      %52 = stablehlo.constant dense<true> : tensor<i1>
      %53 = stablehlo.reduce(%47 init: %52) applies stablehlo.and across dimensions = [0] : (tensor<4xi1>, tensor<i1>) -> tensor<i1>
      %54 = stablehlo.not %53 : tensor<i1>
      %55 = stablehlo.and %51, %54 : tensor<i1>
      stablehlo.return %55 : tensor<i1>
    } do {
      %56 = stablehlo.constant dense<0> : tensor<i32>
      %57 = stablehlo.dynamic_slice %30, %46, %56, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %58 = stablehlo.reshape %57 : (tensor<1x4xf32>) -> tensor<4xf32>
      %59 = stablehlo.constant dense<0> : tensor<i32>
      %60 = stablehlo.dynamic_slice %42, %46, %59, sizes = [1, 4] : (tensor<128x4xf32>, tensor<i32>, tensor<i32>) -> tensor<1x4xf32>
      %61 = stablehlo.reshape %60 : (tensor<1x4xf32>) -> tensor<4xf32>
      %62 = stablehlo.broadcast_in_dim %12, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %63 = stablehlo.multiply %62, %58 : tensor<4xf32>
      %64 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %65 = stablehlo.add %64, %63 : tensor<4xf32>
      %66 = stablehlo.multiply %65, %65 : tensor<4xf32>
      %67 = stablehlo.multiply %66, %65 : tensor<4xf32>
      %68 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %69 = stablehlo.multiply %68, %67 : tensor<4xf32>
      %70 = stablehlo.constant dense<0.5> : tensor<f32>
      %71 = stablehlo.multiply %58, %58 : tensor<4xf32>
      %72 = stablehlo.broadcast_in_dim %70, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %73 = stablehlo.multiply %72, %71 : tensor<4xf32>
      %74 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %75 = stablehlo.multiply %74, %67 : tensor<4xf32>
      %76 = stablehlo.negate %75 : tensor<4xf32>
      %77 = stablehlo.log %67 : tensor<4xf32>
      %78 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %79 = stablehlo.multiply %78, %77 : tensor<4xf32>
      %80 = stablehlo.broadcast_in_dim %8, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %81 = stablehlo.add %73, %80 : tensor<4xf32>
      %82 = stablehlo.add %81, %76 : tensor<4xf32>
      %83 = stablehlo.add %82, %79 : tensor<4xf32>
      %84 = stablehlo.log %61 : tensor<4xf32>
      %85 = stablehlo.compare LT, %84, %83 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %86 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
      %87 = stablehlo.compare GT, %67, %86 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
      %88 = stablehlo.and %85, %87 : tensor<4xi1>
      %89 = stablehlo.select %47, %48, %69 : (tensor<4xi1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
      %90 = stablehlo.or %47, %88 : tensor<4xi1>
      %91 = stablehlo.constant dense<1> : tensor<i32>
      %92 = stablehlo.add %46, %91 : tensor<i32>
      stablehlo.return %92, %90, %89 : tensor<i32>, tensor<4xi1>, tensor<4xf32>
    }
    %93, %94 = stablehlo.rng_bit_generator %31, algorithm =  THREE_FRY : (tensor<2xui64>) -> (tensor<2xui64>, tensor<4xui32>)
    %95 = stablehlo.constant dense<9> : tensor<4xui32>
    %96 = stablehlo.shift_right_logical %94, %95 : tensor<4xui32>
    %97 = stablehlo.convert %96 : (tensor<4xui32>) -> tensor<4xf32>
    %98 = stablehlo.constant dense<1.1920929E-7> : tensor<4xf32>
    %99 = stablehlo.multiply %97, %98 : tensor<4xf32>
    %100 = stablehlo.subtract %3, %2 : tensor<f32>
    %101 = stablehlo.broadcast_in_dim %100, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %102 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %103 = stablehlo.multiply %99, %101 : tensor<4xf32>
    %104 = stablehlo.add %103, %102 : tensor<4xf32>
    %105 = stablehlo.divide %3, %0 : tensor<f32>
    %106 = stablehlo.broadcast_in_dim %105, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %107 = stablehlo.power %104, %106 : tensor<4xf32>
    %108 = stablehlo.broadcast_in_dim %3, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %109 = stablehlo.select %4, %107, %108 : (tensor<i1>, tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %110 = stablehlo.multiply %49#2, %109 : tensor<4xf32>
    %111 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %112 = stablehlo.divide %110, %111 : tensor<4xf32>
    return %112, %93 : tensor<4xf32>, tensor<2xui64>
  }
}
